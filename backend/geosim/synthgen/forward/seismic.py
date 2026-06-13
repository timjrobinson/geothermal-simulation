"""Seismic reflection + microseismic T0 forwards (doc 05 §4 rows 7, 9).

- :class:`SeismicReflectionForward` (``seismic/reflection``) — **degrade-the-truth**
  convolutional model (doc 05 §6 T0): per CMP, sample the truth acoustic impedance
  ``Z = ρ·Vp`` down a vertical column, form the reflectivity series
  ``r = ΔZ/ΣZ`` at layer/fault contacts, convolve with a band-limited **Ricker** wavelet
  (peak ``seis_wavelet_freq``), add band-limited noise → a 2-D zero-offset section. The
  vertical resolution is ``≈ λ/4`` (band-limited = degradation 2): the section "sees"
  the faulted *structure* but is nearly blind to the fluid/temperature field (doc 05
  §4.2 "seismic sees structure, not fluid"). Emits a **SEG-Y** (``segyio``) + a horizons
  **GeoJSON** of the picked strongest reflectors.
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

__all__ = ["SeismicReflectionForward", "MicroseismicForward", "ricker"]


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


class SeismicReflectionForward(T0Forward):
    """T0 reflection: impedance reflectivity ⊛ Ricker → SEG-Y + horizons (doc 05 §4 row7)."""

    method = "seismic"
    submethod = "reflection"

    def simulate(
        self, truth: TruthEarth, acq: Acquisition, rng: np.random.Generator
    ) -> list[Artifact]:
        import segyio

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

        spec = segyio.spec()
        spec.format = 5  # IEEE float32
        spec.samples = (np.arange(ns) * dt * 1000.0)  # ms
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
                f.trace[it] = data[it]

        # horizons GeoJSON (picked strongest reflector per trace, as a LineString)
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

        prov = self._prov(truth, ns=ns, dt=dt, wavelet="ricker",
                          peakFreqHz=acq.seis_wavelet_freq)
        return [
            Artifact(segy_path, "segy", self.method, self.submethod, prov),
            Artifact(gj_path, "geojson", self.method, self.submethod, prov),
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
