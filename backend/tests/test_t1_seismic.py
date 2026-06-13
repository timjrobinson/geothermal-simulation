"""T1 (rigorous) seismic forwards — acoustic reflection + eikonal refraction (doc 05 §4, §6 T1).

Two rigorous seismic forwards behind the uniform :class:`ForwardModel` contract:

- :class:`SeismicReflectionRigorousForward` (``fidelity="rigorous"``) — a proper
  acoustic/convolutional synthetic from the FULL truth impedance series, depth→TWT with
  the true velocity, Ricker convolution, multiples + band-limited noise; emits the SAME
  SEG-Y + horizons GeoJSON the T0 does (doc 05 §4 rigorous column).
- :class:`SeismicRefractionRigorousForward` (``seismic/refraction``) — first-break
  traveltimes from pykonal's eikonal solver through the truth ``Vp`` model; emits SEG-Y +
  a CSV of picks.

These tests assert, on a SMALL truth earth: registry wiring (fidelity-aware, T0 untouched);
the rigorous reflection synthetic runs + the SEG-Y re-reads through the doc-03 adapter with
reflection energy at the true impedance contrasts; the pykonal refraction traveltimes are
monotone non-decreasing in offset and physically sensible (increasing apparent velocity);
and the T0 reflection forward still works.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from geosim.ingestion import RawSource
from geosim.ingestion.adapters.seismic import SeismicSegyAdapter
from geosim.synthgen import (
    AnomalySpec,
    FaultSpec,
    FrameSpec,
    GeothermSpec,
    LayerSpec,
    SceneSpec,
    SurfaceSpec,
    compile_scene,
)
from geosim.synthgen.forward import (
    RIGOROUS_FORWARD_MODELS,
    Acquisition,
    Artifact,
    ForwardModel,
    SeismicReflectionForward,
    SeismicReflectionRigorousForward,
    SeismicRefractionRigorousForward,
    get_forward,
)
from geosim.synthgen.forward.base import sample_volume_at, world_axes
from geosim.synthgen.forward.seismic import _eikonal_traveltimes, _seismic_line

# --------------------------------------------------------------- a tiny truth earth


def _tiny_scene(seed: int = 7) -> SceneSpec:
    """A SMALL layered scene with sharp impedance contrasts at the layer boundaries."""
    return SceneSpec(
        id="tiny-t1-seismic-v1",
        seed=seed,
        frame=FrameSpec(
            xmin=-400, xmax=400, ymin=-400, ymax=400,
            zmin=-800, zmax=200, dx=100, dy=100, dz=100,
        ),
        surface=SurfaceSpec(kind="flat", base_elev=100.0),
        layers=(
            LayerSpec("alluvium", "surface", (80.0, 140.0)),
            LayerSpec("volcanics", "conformable", (150.0, 250.0)),
            LayerSpec("basement_granite", "conformable", "fill"),
        ),
        faults=(
            FaultSpec("range-front", trace=((-400, -100), (400, 100)),
                      kind="normal", dip=60, dip_azimuth=90, throw=150, is_conduit=True),
        ),
        geotherm=GeothermSpec(surface_temp=15.0, gradient=45.0),
        anomalies=(
            AnomalySpec(
                "upflow", footprint_center=(0.0, 0.0), footprint_radius_xy=200.0,
                top_elev=100.0, bottom_elev=-600.0, controlled_by="range-front",
                temp_peak=200.0, alteration_frac=0.8, porosity_boost=0.04,
                salinity_tds=8000.0, fracture_density=0.5,
            ),
        ),
        rock_physics="default-v1",
    )


@pytest.fixture(scope="module")
def earth():
    return compile_scene(_tiny_scene())


@pytest.fixture()
def acq(tmp_path):
    # few traces, but enough samples that the deep reflectors fall inside the record.
    return Acquisition(
        seis_n_traces=12, seis_n_samples=512, params={"out_dir": str(tmp_path)}
    )


def _rng():
    return np.random.default_rng(99)


def _read_segy(art: Artifact):
    return SeismicSegyAdapter().parse(
        RawSource(filename=art.path.name, path=str(art.path))
    )


# --------------------------------------------------------------- registry wiring


def test_rigorous_reflection_registered_and_conforms():
    fwd = get_forward("seismic", "reflection", fidelity="rigorous")
    assert isinstance(fwd, SeismicReflectionRigorousForward)
    assert isinstance(fwd, ForwardModel)
    assert fwd.fidelity == "rigorous"
    assert fwd.method == "seismic" and fwd.submethod == "reflection"
    assert ("seismic", "reflection") in RIGOROUS_FORWARD_MODELS


def test_rigorous_refraction_registered_and_conforms():
    fwd = get_forward("seismic", "refraction", fidelity="rigorous")
    assert isinstance(fwd, SeismicRefractionRigorousForward)
    assert isinstance(fwd, ForwardModel)
    assert fwd.fidelity == "rigorous"
    assert fwd.method == "seismic" and fwd.submethod == "refraction"
    assert ("seismic", "refraction") in RIGOROUS_FORWARD_MODELS


def test_plausible_default_unchanged():
    # default fidelity is still T0; the rigorous selector does not swap it silently.
    assert isinstance(get_forward("seismic", "reflection"), SeismicReflectionForward)
    assert not isinstance(
        get_forward("seismic", "reflection"), SeismicReflectionRigorousForward
    )
    # refraction has no T0 tier → plausible lookup raises (rigorous-only submethod).
    with pytest.raises(KeyError):
        get_forward("seismic", "refraction", fidelity="plausible")


# --------------------------------------------------------------- rigorous reflection


def test_rigorous_reflection_emits_same_native_files_and_re_reads(earth, acq):
    """T1 reflection emits the SAME SEG-Y section + horizons GeoJSON the T0 does."""
    arts = SeismicReflectionRigorousForward().simulate(earth, acq, _rng())
    assert {a.fmt for a in arts} == {"segy", "geojson"}
    segy = next(a for a in arts if a.fmt == "segy")
    assert segy.path.suffix == ".segy"

    res = _read_segy(segy)
    assert not res.warnings or all(w.severity.value != "high" for w in res.warnings)
    pm = res.property_models[0]
    assert pm.support == "section"
    assert pm.property == "velocity_p"
    assert pm.values.shape == (acq.seis_n_samples, acq.seis_n_traces)
    assert np.all(np.isfinite(pm.values))
    # horizons joined as a horizon feature (doc 03 §2 surfaces).
    assert any(f.feature_type == "horizon" for f in res.features)

    prov = segy.provenance.to_dict()
    assert prov["source"] == "synthgen"
    assert prov["fidelity"] == "rigorous"
    assert prov["sceneId"] == earth.spec.id
    assert prov["engine"] == "acoustic-convolutional"


def test_rigorous_reflection_energy_at_true_impedance_contrasts(earth, acq):
    """Reflection energy concentrates at the two-way times of the true impedance jumps.

    Recompute the truth impedance column under a central trace, find the depth sample with
    the largest reflection coefficient, map it to its true TWT, and assert the re-read
    section has a band-limited amplitude peak within a wavelet length of that time — i.e.
    the reflectors sit at the genuine ρ·Vp contrasts, not at an arbitrary time.
    """
    arts = SeismicReflectionRigorousForward().simulate(earth, acq, _rng())
    segy = next(a for a in arts if a.fmt == "segy")
    pm = _read_segy(segy).property_models[0].values  # (n_samples, n_traces)

    z, y, x = world_axes(earth)
    axes = (z, y, x)
    dz = earth.spacing[0]
    rho = earth.property_volume("density").astype(np.float64)
    vp = earth.property_volume("velocity_p").astype(np.float64)
    imp_vol = rho * vp

    cx, cy = _seismic_line(earth, acq)
    it = acq.seis_n_traces // 2
    zsorted = np.sort(z)[::-1]
    pts = np.column_stack(
        [zsorted, np.full_like(zsorted, cy[it]), np.full_like(zsorted, cx[it])]
    )
    imp = sample_volume_at(imp_vol, axes, pts)
    vpc = sample_volume_at(vp, axes, pts)
    refl = np.zeros_like(imp)
    denom = imp[1:] + imp[:-1]
    refl[1:] = np.where(denom > 0.0, (imp[1:] - imp[:-1]) / denom, 0.0)
    twt = np.cumsum(2.0 * dz / np.maximum(vpc, 1.0))
    dt = acq.seis_dt
    ns = acq.seis_n_samples
    # strongest reflection coefficient whose two-way time falls inside the record.
    k_samp = np.round(twt / dt).astype(int)
    in_record = k_samp < ns
    assert np.any(in_record & (np.abs(refl) > 0)), "no reflector inside the record"
    refl_in = np.where(in_record, np.abs(refl), 0.0)
    k_true = int(k_samp[np.argmax(refl_in)])

    # band-limited amplitude envelope of the re-read trace.
    trace = np.abs(pm[:, it])
    k_obs = int(np.argmax(trace))
    # the strongest reflection peak is within ~one Ricker half-width of the true contrast.
    assert abs(k_obs - k_true) <= 12
    # and there IS a real reflection (not just noise): peak well above the median.
    assert trace[k_obs] > 5.0 * np.median(trace)


def test_rigorous_reflection_is_deterministic(earth, tmp_path):
    from dataclasses import replace

    base = Acquisition(seis_n_traces=12, seis_n_samples=512)
    a = SeismicReflectionRigorousForward().simulate(
        earth, replace(base, params={"out_dir": str(tmp_path / "a")}),
        np.random.default_rng(5),
    )
    b = SeismicReflectionRigorousForward().simulate(
        earth, replace(base, params={"out_dir": str(tmp_path / "b")}),
        np.random.default_rng(5),
    )
    va = _read_segy(next(x for x in a if x.fmt == "segy")).property_models[0].values
    vb = _read_segy(next(x for x in b if x.fmt == "segy")).property_models[0].values
    np.testing.assert_allclose(va, vb)


def test_t0_reflection_still_works(earth, acq):
    """The T0 reflection forward is unchanged and still round-trips (doc 05 §6)."""
    arts = SeismicReflectionForward().simulate(earth, acq, _rng())
    assert {a.fmt for a in arts} == {"segy", "geojson"}
    segy = next(a for a in arts if a.fmt == "segy")
    pm = _read_segy(segy).property_models[0]
    assert pm.values.shape == (acq.seis_n_samples, acq.seis_n_traces)
    assert segy.provenance.fidelity == "plausible"


# --------------------------------------------------------------- rigorous refraction


def test_refraction_emits_segy_and_picks_csv(earth, acq):
    """The pykonal refraction forward emits a SEG-Y + a first-break picks CSV."""
    arts = SeismicRefractionRigorousForward().simulate(earth, acq, _rng())
    assert {a.fmt for a in arts} == {"segy", "csv"}

    segy = next(a for a in arts if a.fmt == "segy")
    res = _read_segy(segy)
    pm = res.property_models[0]
    assert pm.values.shape == (acq.seis_n_samples, acq.seis_n_traces)
    assert np.all(np.isfinite(pm.values))

    prov = segy.provenance.to_dict()
    assert prov["fidelity"] == "rigorous"
    assert prov["engine"] == "pykonal"
    assert prov["solver"] == "eikonal"


def _read_picks(art: Artifact):
    with open(art.path, newline="") as f:
        rows = list(csv.DictReader(f))
    offsets = np.array([float(r["offset_m"]) for r in rows])
    tt = np.array([float(r["traveltime_s"]) for r in rows])
    return offsets, tt


def test_refraction_traveltimes_monotonic_in_offset(earth, acq):
    """First-break traveltimes are monotone non-decreasing with source-receiver offset."""
    arts = SeismicRefractionRigorousForward().simulate(earth, acq, _rng())
    csv_art = next(a for a in arts if a.fmt == "csv")
    offsets, tt = _read_picks(csv_art)

    # offsets sorted ascending (shot at the start of the spread).
    assert np.all(np.diff(offsets) >= -1e-6)
    # the zero-offset shot has zero traveltime.
    assert tt[0] == pytest.approx(0.0, abs=1e-9)
    # traveltime never decreases as offset grows (eikonal first arrival).
    assert np.all(np.diff(tt) >= -1e-9)
    # and it strictly grows over the full spread (real propagation, not a flat field).
    assert tt[-1] > tt[1] > 0.0


def test_refraction_traveltimes_physically_sensible(earth, acq):
    """Apparent velocity is sensible and increases with offset (layered head wave).

    The near-offset slope is the (slow) shallow direct-wave velocity; at far offsets the
    head wave through the faster basement overtakes it, so the apparent velocity
    (offset/time) increases — the diagnostic of a refraction spread over a layered earth.
    """
    arts = SeismicRefractionRigorousForward().simulate(earth, acq, _rng())
    offsets, tt = _read_picks(next(a for a in arts if a.fmt == "csv"))

    # apparent velocity between consecutive picks stays within plausible crustal bounds.
    d_off = np.diff(offsets)
    d_t = np.diff(tt)
    good = (d_off > 1.0) & (d_t > 1e-6)
    v_app = d_off[good] / d_t[good]
    assert np.all(v_app > 300.0)      # nothing slower than weathered soil
    assert np.all(v_app < 9000.0)     # nothing faster than the fastest crustal rock

    # apparent velocity to the farthest geophone exceeds that to a near one (head wave).
    n = offsets.size
    v_near = offsets[max(n // 4, 1)] / max(tt[max(n // 4, 1)], 1e-9)
    v_far = offsets[-1] / max(tt[-1], 1e-9)
    assert v_far >= v_near - 1e-6


def test_refraction_traveltimes_match_eikonal_solver(earth, acq):
    """The emitted picks equal the raw pykonal eikonal solution (no hidden fudge)."""
    cx, cy = _seismic_line(earth, acq)
    geo = np.column_stack([cx, cy])
    tt_direct = _eikonal_traveltimes(earth, (float(cx[0]), float(cy[0])), geo)

    arts = SeismicRefractionRigorousForward().simulate(earth, acq, _rng())
    _, tt_csv = _read_picks(next(a for a in arts if a.fmt == "csv"))
    np.testing.assert_allclose(tt_csv, tt_direct, atol=1e-6)
