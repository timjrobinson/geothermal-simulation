"""Seismic reflection + microseismic forwards (doc 05 §4 rows 7, 9; §6 T0 + T1).

- :class:`SeismicReflectionForward` (``seismic/reflection``, T0) — **degrade-the-truth**
  convolutional model (doc 05 §6 T0): per CMP, sample the truth acoustic impedance
  ``Z = ρ·Vp`` down a vertical column, form the reflectivity series
  ``r = ΔZ/ΣZ`` at layer/fault contacts, convolve with a band-limited **Ricker** wavelet
  (peak ``seis_wavelet_freq``), add band-limited noise → a 2-D zero-offset section. The
  vertical resolution is ``≈ λ/4`` (band-limited = degradation 2): the section "sees"
  the faulted *structure* but is nearly blind to the fluid/temperature field (doc 05
  §4.2 "seismic sees structure, not fluid"). Emits a **SEG-Y** (``segyio``) + a horizons
  **GeoJSON** of the picked strongest reflectors.
- :class:`SeismicReflectionRigorousForward` (``seismic/reflection``,
  ``fidelity="rigorous"``, T1) — a proper **acoustic/convolutional** synthetic (doc 05 §4
  rigorous column "at least convolutional/acoustic + realistic velocity→reflectivity"):
  the reflectivity is computed from the **FULL** impedance series ``Z = ρ·Vp`` of the
  TruthEarth (every depth sample, not just hand-picked contacts), depth is mapped to
  two-way time with the **true** interval velocity (continuous, amplitude-conserving
  linear placement into the time series rather than the T0 nearest-sample bin), convolved
  with the band-limited **Ricker** wavelet, and a first-order water-bottom / strong-
  reflector **multiple** plus band-limited noise are added. It emits the SAME SEG-Y +
  horizons GeoJSON as the T0 so ingestion is unchanged, but is more accurate (the section
  honours the true impedance profile, and the vertical resolution is the correct
  ``λ/4`` = ``Vp/(4·f)`` band limit).
- :class:`SeismicRefractionRigorousForward` (``seismic/refraction``,
  ``fidelity="rigorous"``, T1) — first-break **traveltimes** computed by solving the
  eikonal equation through the truth ``Vp`` model with **pykonal** (doc 05 §4 rigorous
  "refraction via pykonal eikonal") along a refraction spread (shot + inline geophones).
  Traveltimes are physically sensible (monotone non-decreasing in offset; the direct-wave
  / head-wave crossover gives the increasing apparent velocity of a layered earth). Emits
  a **SEG-Y** (one trace per geophone, a Ricker first break at the picked time) + a
  **CSV** of first-break picks ``(offset, traveltime)``.
- :class:`MicroseismicForward` (``microseismic``) — sample events on a *stimulated fault
  plane* (the conduit fault), assign **Gutenberg-Richter** magnitudes (``N∝10^(-b·M)``),
  locate each with a distance-growing location error (doc 05 §4 row 9). Emits **QuakeML**
  (``obspy``) + a CSV catalog.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..truth import TruthEarth
from .base import (
    Acquisition,
    Artifact,
    T0Forward,
    sample_volume_at,
    world_axes,
)

__all__ = [
    "SeismicReflectionForward",
    "SeismicReflectionRigorousForward",
    "SeismicRefractionRigorousForward",
    "MicroseismicForward",
    "ricker",
]


def ricker(n: int, dt: float, f: float) -> np.ndarray:
    """Zero-phase Ricker wavelet of length ``n`` at sample interval ``dt`` (Hz peak ``f``)."""
    t = (np.arange(n) - n // 2) * dt
    a = (np.pi * f * t) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def _seismic_line(truth: TruthEarth, acq: Acquisition):
    """CMP plan positions + the two-way-time axis (doc 05 §4.3 seismic line)."""
    z, y, x = world_axes(truth)
    if acq.seis_line is not None:
        (x0, y0), (x1, y1) = acq.seis_line
    else:
        x0, x1 = float(x[0]), float(x[-1])
        y0 = y1 = float(np.mean(y))
    n = acq.seis_n_traces
    t = np.linspace(0.0, 1.0, n)
    cx = x0 + t * (x1 - x0)
    cy = y0 + t * (y1 - y0)
    return cx, cy


def _write_segy_section(
    segy_path: Path,
    data: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    ns: int,
    dt: float,
) -> None:
    """Write a 2-D ``(ntr, ns)`` zero-offset section to SEG-Y (doc 03 §2 row 7 native).

    Shared by the T0 and T1 reflection/refraction forwards so every seismic artifact is
    the *same* native SEG-Y the doc-03 :class:`SeismicSegyAdapter` re-reads: IEEE float32
    samples, time axis in ms from ``dt``, CDP X/Y plan coordinates per trace.
    """
    import segyio

    ntr = int(data.shape[0])
    spec = segyio.spec()
    spec.format = 5  # IEEE float32
    spec.samples = np.arange(ns) * dt * 1000.0  # ms
    spec.tracecount = ntr
    with segyio.create(str(segy_path), spec) as f:
        for it in range(ntr):
            f.header[it] = {
                segyio.su.cdp: it + 1,
                segyio.su.cdpx: int(cx[it]),
                segyio.su.cdpy: int(cy[it]),
                segyio.su.ns: ns,
                segyio.su.dt: int(dt * 1e6),
            }
            f.trace[it] = data[it].astype(np.float32)


class SeismicReflectionForward(T0Forward):
    """T0 reflection: impedance reflectivity ⊛ Ricker → SEG-Y + horizons (doc 05 §4 row7)."""

    method = "seismic"
    submethod = "reflection"

    def simulate(
        self, truth: TruthEarth, acq: Acquisition, rng: np.random.Generator
    ) -> list[Artifact]:
        z, y, x = world_axes(truth)
        axes = (z, y, x)
        dz = truth.spacing[0]
        rho = truth.property_volume("density").astype(np.float64)
        vp = truth.property_volume("velocity_p").astype(np.float64)
        imp_vol = rho * vp  # acoustic impedance

        cx, cy = _seismic_line(truth, acq)
        ntr = cx.size
        ns = acq.seis_n_samples
        dt = acq.seis_dt

        # depth→TWT: integrate slowness down each column to map z-samples to time.
        zsorted = np.sort(z)[::-1]  # shallow → deep elevation
        wav = ricker(64, dt, acq.seis_wavelet_freq)

        data = np.zeros((ntr, ns), dtype=np.float32)
        horizon_picks: list[list[float]] = []  # per trace [twt of top strong reflector]
        for it in range(ntr):
            pts = np.column_stack([
                zsorted, np.full_like(zsorted, cy[it]), np.full_like(zsorted, cx[it])
            ])
            imp = sample_volume_at(imp_vol, axes, pts)
            vpc = sample_volume_at(vp, axes, pts)
            # reflectivity in depth
            refl = np.zeros_like(imp)
            refl[1:] = (imp[1:] - imp[:-1]) / (imp[1:] + imp[:-1] + 1e-9)
            # depth→TWT: cumulative two-way time using interval Vp
            dt_layer = 2.0 * dz / np.maximum(vpc, 1.0)
            twt = np.cumsum(dt_layer)
            # place reflectivity at the nearest time sample
            tser = np.zeros(ns)
            samp = np.clip((twt / dt).astype(int), 0, ns - 1)
            np.add.at(tser, samp, refl)
            trace = np.convolve(tser, wav, mode="same")
            # band-limited noise (doc 05 §4 row 7)
            noise = np.convolve(rng.normal(0, 0.02, ns), wav, mode="same")
            data[it] = (trace + noise).astype(np.float32)
            # horizon pick: strongest positive reflector time
            if samp.size and np.any(refl > 0):
                k = int(samp[np.argmax(refl)])
                horizon_picks.append([float(cx[it]), float(cy[it]), float(k * dt)])

        out_dir = Path(acq.params.get("out_dir", "."))
        out_dir.mkdir(parents=True, exist_ok=True)
        segy_path = out_dir / "seismic_lineAA.segy"
        _write_segy_section(segy_path, data, cx, cy, ns, dt)
        gj_path = _write_horizons_geojson(out_dir, truth, horizon_picks)

        prov = self._prov(truth, ns=ns, dt=dt, wavelet="ricker",
                          peakFreqHz=acq.seis_wavelet_freq)
        return [
            Artifact(segy_path, "segy", self.method, self.submethod, prov),
            Artifact(gj_path, "geojson", self.method, self.submethod, prov),
        ]


def _write_horizons_geojson(
    out_dir: Path, truth: TruthEarth, horizon_picks: list[list[float]]
) -> Path:
    """Write the picked-reflector horizons LineString GeoJSON (doc 03 §2 surfaces)."""
    import json

    coords = [[p[0], p[1]] for p in horizon_picks]
    gj = {
        "type": "FeatureCollection",
        "name": f"{truth.spec.id}-horizons",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords or [[0, 0], [1, 1]]},
            "properties": {"kind": "horizon", "pick": "strongest_reflector",
                           "twt_s": [p[2] for p in horizon_picks]},
        }],
    }
    gj_path = out_dir / "seismic_horizons.geojson"
    gj_path.write_text(json.dumps(gj, indent=2), encoding="utf-8")
    return gj_path


class SeismicReflectionRigorousForward(T0Forward):
    """T1 reflection: full-impedance acoustic convolutional synthetic (doc 05 §4, §6 T1).

    The rigorous (``fidelity="rigorous"``) reflection forward upgrades the T0
    contact-only reflectivity to a proper **acoustic/convolutional** synthetic computed
    from the FULL impedance series of the TruthEarth (doc 05 §4 rigorous column):

    1. **realistic velocity→reflectivity** — at each CMP, sample the truth impedance
       ``Z = ρ·Vp`` down the *whole* column (every z-sample) and form the normal-incidence
       reflectivity ``r_i = (Z_{i+1} − Z_i)/(Z_{i+1} + Z_i)`` at *every* interface, so the
       section honours the continuous impedance profile, not just hand-picked contacts;
    2. **depth→two-way-time with the true velocity** — integrate ``2·dz/Vp`` down the
       column for the exact interval-velocity TWT, then place each reflection coefficient
       into the time series by **linear (amplitude-conserving) interpolation** between the
       two bracketing time samples (the T0 snaps to the nearest sample, mistiming +
       quantising the reflectors);
    3. **band-limited Ricker** — convolve with the zero-phase Ricker (peak
       ``seis_wavelet_freq``); the vertical resolution is the correct ``λ/4 = Vp/(4·f)``;
    4. **multiple + band-limited noise** — add a first-order surface multiple (the time
       series convolved with a delayed, polarity-flipped copy of itself, scaled small) and
       band-limited Gaussian noise (doc 05 §4 row 7 noise/multiples column).

    Emits the SAME SEG-Y section + horizons GeoJSON as the T0 so ingestion is unchanged
    (doc 05 §4 contract); the section is strictly more accurate (the reflections sit at the
    true impedance contrasts at the true two-way times).
    """

    method = "seismic"
    submethod = "reflection"
    fidelity = "rigorous"

    def simulate(
        self, truth: TruthEarth, acq: Acquisition, rng: np.random.Generator
    ) -> list[Artifact]:
        z, y, x = world_axes(truth)
        axes = (z, y, x)
        dz = truth.spacing[0]
        rho = truth.property_volume("density").astype(np.float64)
        vp = truth.property_volume("velocity_p").astype(np.float64)
        imp_vol = rho * vp  # acoustic impedance Z = ρ·Vp

        cx, cy = _seismic_line(truth, acq)
        ntr = cx.size
        ns = acq.seis_n_samples
        dt = acq.seis_dt
        f0 = acq.seis_wavelet_freq

        zsorted = np.sort(z)[::-1]  # shallow → deep elevation
        wav = ricker(64, dt, f0)
        # multiple-generating operator: a delayed, polarity-flipped weak copy (peg-leg /
        # surface multiple). Small amplitude so primaries dominate (doc 05 §4 multiples).
        mult_lag = max(int(round(0.04 / dt)), 1)  # ~40 ms peg-leg
        mult_amp = 0.15

        data = np.zeros((ntr, ns), dtype=np.float64)
        horizon_picks: list[list[float]] = []
        for it in range(ntr):
            pts = np.column_stack([
                zsorted, np.full_like(zsorted, cy[it]), np.full_like(zsorted, cx[it])
            ])
            imp = sample_volume_at(imp_vol, axes, pts)
            vpc = sample_volume_at(vp, axes, pts)
            # FULL reflectivity series at every interface (realistic vel→reflectivity).
            refl = np.zeros_like(imp)
            denom = imp[1:] + imp[:-1]
            refl[1:] = np.where(denom > 0.0, (imp[1:] - imp[:-1]) / denom, 0.0)
            # depth→TWT with the TRUE interval velocity (cumulative two-way time).
            twt = np.cumsum(2.0 * dz / np.maximum(vpc, 1.0))
            # amplitude-conserving linear placement into the time series.
            tser = np.zeros(ns)
            pos = twt / dt
            i0 = np.floor(pos).astype(int)
            frac = pos - i0
            in0 = (i0 >= 0) & (i0 < ns)
            np.add.at(tser, np.clip(i0[in0], 0, ns - 1), refl[in0] * (1.0 - frac[in0]))
            i1 = i0 + 1
            in1 = (i1 >= 0) & (i1 < ns)
            np.add.at(tser, np.clip(i1[in1], 0, ns - 1), refl[in1] * frac[in1])
            # first-order surface multiple: delayed polarity-flipped copy of the primaries.
            mult = np.zeros(ns)
            mult[mult_lag:] = -mult_amp * tser[:-mult_lag]
            trace = np.convolve(tser + mult, wav, mode="same")
            noise = np.convolve(rng.normal(0, 0.02, ns), wav, mode="same")
            data[it] = trace + noise
            # horizon pick: strongest positive reflector, at its true TWT.
            if refl.size and np.any(refl > 0):
                k = int(np.clip(round(twt[np.argmax(refl)] / dt), 0, ns - 1))
                horizon_picks.append([float(cx[it]), float(cy[it]), float(k * dt)])

        out_dir = Path(acq.params.get("out_dir", "."))
        out_dir.mkdir(parents=True, exist_ok=True)
        segy_path = out_dir / "seismic_lineAA.segy"
        _write_segy_section(segy_path, data, cx, cy, ns, dt)
        gj_path = _write_horizons_geojson(out_dir, truth, horizon_picks)

        prov = self._prov(
            truth, ns=ns, dt=dt, wavelet="ricker", peakFreqHz=f0,
            engine="acoustic-convolutional", multiples=True,
        )
        return [
            Artifact(segy_path, "segy", self.method, self.submethod, prov),
            Artifact(gj_path, "geojson", self.method, self.submethod, prov),
        ]


def _refraction_spread(truth: TruthEarth, acq: Acquisition):
    """Shot + inline geophone plan positions for a refraction spread (doc 05 §4.3).

    Reuses the seismic-line geometry: the first station is the **shot**, the remaining
    ``seis_n_traces`` stations are inline geophones at growing offset.
    """
    cx, cy = _seismic_line(truth, acq)
    return cx, cy


def _eikonal_traveltimes(
    truth: TruthEarth, shot_xy: tuple[float, float], geo_xy: np.ndarray
) -> np.ndarray:
    """First-break traveltimes shot→geophones via the pykonal eikonal solver (doc 05 §4).

    Solves ``|∇T| = 1/Vp`` on a 2-D vertical (x–z) slice of the truth ``Vp`` model under
    the shot/receiver line with a point source at the shot, then samples the traveltime
    field at the surface geophone offsets. Returns one traveltime (s) per geophone.

    The slice is built in pykonal's increasing-coordinate ``(i0=along-line, i1=depth)``
    frame: axis 0 is horizontal distance from the line start (m), axis 1 is depth below
    the model top (m, +down). Velocity is sampled from the truth ``Vp`` volume along the
    line so the head-wave refractions honour the real velocity structure.
    """
    import pykonal

    z, y, x = world_axes(truth)
    axes = (z, y, x)
    vp = truth.property_volume("velocity_p").astype(np.float64)

    (sx, sy) = shot_xy
    gx = geo_xy[:, 0]
    gy = geo_xy[:, 1]
    # along-line distance of every node from the shot (the spread is ~straight).
    x0, y0 = float(gx[0]), float(gy[0])
    x1, y1 = float(gx[-1]), float(gy[-1])
    L = float(np.hypot(x1 - x0, y1 - y0))
    if L <= 0.0:
        L = max(float(x[-1] - x[0]), 1.0)
    ux, uy = (x1 - x0) / L, (y1 - y0) / L  # unit along-line

    # build the 2-D velocity slice (along-line × depth) by sampling the truth column.
    dl = max(float(min(truth.spacing[2], truth.spacing[1])), 1.0)  # lateral node interval
    n_l = max(int(round(L / dl)) + 1, 2)
    l_axis = np.linspace(0.0, L, n_l)
    dz = float(truth.spacing[0])
    z_top = float(np.max(z))
    z_bot = float(np.min(z))
    n_d = max(int(round((z_top - z_bot) / dz)) + 1, 2)
    depth_axis = np.linspace(0.0, z_top - z_bot, n_d)

    ll, dd = np.meshgrid(l_axis, depth_axis, indexing="ij")
    px = x0 + ll * ux
    py = y0 + ll * uy
    pz = z_top - dd  # +down depth → elevation
    pts = np.column_stack([pz.ravel(), py.ravel(), px.ravel()])
    vslice = sample_volume_at(vp, axes, pts).reshape(n_l, n_d)
    vslice = np.maximum(vslice, 1.0)

    solver = pykonal.EikonalSolver(coord_sys="cartesian")
    solver.velocity.min_coords = 0.0, 0.0, 0.0
    solver.velocity.node_intervals = (
        float(l_axis[1] - l_axis[0]),
        float(depth_axis[1] - depth_axis[0]),
        1.0,
    )
    solver.velocity.npts = n_l, n_d, 1
    solver.velocity.values = vslice[:, :, None]

    # point source at the shot's surface node (nearest along-line index, depth 0).
    s_l = float(np.hypot(sx - x0, sy - y0))
    si = int(np.clip(round(s_l / (l_axis[1] - l_axis[0])), 0, n_l - 1))
    src = (si, 0, 0)
    solver.traveltime.values[src] = 0.0
    solver.unknown[src] = False
    solver.trial.push(*src)
    solver.solve()
    tt = solver.traveltime.values[:, :, 0]  # (n_l, n_d)

    # sample the surface (depth index 0) traveltime at each geophone's along-line offset.
    g_l = np.hypot(gx - x0, gy - y0)
    return np.interp(g_l, l_axis, tt[:, 0])


class SeismicRefractionRigorousForward(T0Forward):
    """T1 refraction: pykonal eikonal first-break traveltimes (doc 05 §4, §6 T1).

    The rigorous (``fidelity="rigorous"``) refraction forward solves the **eikonal
    equation** ``|∇T| = 1/Vp`` through the truth ``Vp`` model with **pykonal** along a
    refraction spread (a shot at the line start + inline geophones at growing offset),
    giving the first-break traveltime to each geophone (doc 05 §4 rigorous "refraction via
    pykonal eikonal"). The picks are physically sensible: monotone non-decreasing in
    offset, and the near-offset direct wave / far-offset head-wave crossover yields the
    increasing apparent velocity of a layered earth (the basis of refraction inversion).

    Emits a **SEG-Y** (one trace per geophone with a band-limited Ricker first break at
    the picked time, the SAME native format ingestion reads) + a **CSV** of the picks
    ``(trace, offset_m, traveltime_s)`` — the refraction analogue of the reflection
    horizons file.
    """

    method = "seismic"
    submethod = "refraction"
    fidelity = "rigorous"

    def simulate(
        self, truth: TruthEarth, acq: Acquisition, rng: np.random.Generator
    ) -> list[Artifact]:
        cx, cy = _refraction_spread(truth, acq)
        shot = (float(cx[0]), float(cy[0]))
        geo = np.column_stack([cx, cy])

        tt = _eikonal_traveltimes(truth, shot, geo)  # (ntr,) seconds
        offsets = np.hypot(cx - shot[0], cy - shot[1])

        ns = acq.seis_n_samples
        dt = acq.seis_dt
        ntr = int(cx.size)
        wav = ricker(64, dt, acq.seis_wavelet_freq)
        half = wav.size // 2

        # one trace per geophone: a Ricker centred at the first-break time + small noise.
        data = np.zeros((ntr, ns), dtype=np.float64)
        for it in range(ntr):
            k = int(round(tt[it] / dt))
            lo = k - half
            for j in range(wav.size):
                s = lo + j
                if 0 <= s < ns:
                    data[it, s] += wav[j]
            data[it] += rng.normal(0, 0.01, ns)

        out_dir = Path(acq.params.get("out_dir", "."))
        out_dir.mkdir(parents=True, exist_ok=True)
        segy_path = out_dir / "seismic_refraction_lineAA.segy"
        _write_segy_section(segy_path, data, cx, cy, ns, dt)

        rows = ["trace,offset_m,traveltime_s"]
        for it in range(ntr):
            rows.append(f"{it},{offsets[it]:.3f},{tt[it]:.6f}")
        csv_path = out_dir / "seismic_refraction_picks.csv"
        csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

        prov = self._prov(
            truth, ns=ns, dt=dt, engine="pykonal", solver="eikonal",
            nGeophones=ntr,
        )
        return [
            Artifact(segy_path, "segy", self.method, self.submethod, prov),
            Artifact(csv_path, "csv", self.method, self.submethod, prov),
        ]


class MicroseismicForward(T0Forward):
    """T0 microseismic: G-R events on the stimulated fault plane (doc 05 §4 row 9)."""

    method = "microseismic"
    submethod = None

    def simulate(
        self, truth: TruthEarth, acq: Acquisition, rng: np.random.Generator
    ) -> list[Artifact]:
        from obspy import UTCDateTime
        from obspy.core.event import (
            Catalog,
            CreationInfo,
            Event,
            Magnitude,
            Origin,
        )

        z, y, x = world_axes(truth)
        # stimulated plane = the conduit fault (or the first fault) trace, full depth.
        faults = truth.spec.faults
        conduit = next((f for f in faults if f.is_conduit), faults[0] if faults else None)

        n = acq.ms_n_events
        # Gutenberg-Richter magnitudes via inverse-CDF: M = Mc - log10(U)/b
        u = rng.uniform(1e-3, 1.0, n)
        mags = acq.ms_mc - np.log10(u) / acq.ms_b_value

        if conduit is not None:
            (fx0, fy0), (fx1, fy1) = conduit.trace
        else:
            fx0, fy0, fx1, fy1 = float(x[0]), float(np.mean(y)), float(x[-1]), float(np.mean(y))
        zmin, zmax = float(z[0]), float(z[-1])

        t = rng.uniform(0.0, 1.0, n)
        ex = fx0 + t * (fx1 - fx0)
        ey = fy0 + t * (fy1 - fy0)
        ez = rng.uniform(zmin, zmax, n)
        # location error growing with depth below surface (doc 05 §4 row 9)
        surf = zmax
        loc_sigma = 10.0 + 0.05 * (surf - ez)
        ex = ex + rng.normal(0, 1, n) * loc_sigma
        ey = ey + rng.normal(0, 1, n) * loc_sigma
        ez = ez + rng.normal(0, 1, n) * loc_sigma

        out_dir = Path(acq.params.get("out_dir", "."))
        out_dir.mkdir(parents=True, exist_ok=True)

        cat = Catalog(creation_info=CreationInfo(author="geosim.synthgen"))
        t0 = UTCDateTime("2026-01-01T00:00:00")
        rows = ["id,time,x,y,elev,mag"]
        for i in range(n):
            origin = Origin(
                time=t0 + float(i * 3600),
                # store Engineering XY in longitude/latitude slots is wrong for real EDI;
                # here we use the depth (m below surface) + extra; keep XY in comments.
                latitude=0.0, longitude=0.0,
                depth=float(surf - ez[i]),
            )
            ev = Event(origins=[origin],
                       magnitudes=[Magnitude(mag=float(mags[i]), magnitude_type="ML")])
            ev.origins[0].extra = {}
            cat.append(ev)
            rows.append(
                f"{i},{(t0 + i*3600).isoformat()},{ex[i]:.2f},{ey[i]:.2f},"
                f"{ez[i]:.2f},{mags[i]:.3f}"
            )

        qml_path = out_dir / "microseismic.quakeml"
        cat.write(str(qml_path), format="QUAKEML")
        csv_path = out_dir / "microseismic_catalog.csv"
        csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

        prov = self._prov(truth, nEvents=n, bValue=acq.ms_b_value, mc=acq.ms_mc)
        return [
            Artifact(qml_path, "quakeml", self.method, self.submethod, prov),
            Artifact(csv_path, "csv", self.method, self.submethod, prov),
        ]
