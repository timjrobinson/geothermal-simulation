"""T1 (rigorous) MT forward — exact 1-D layered plane-wave impedance (doc 05 §4, §6 T1).

The rigorous MT forward (:class:`MTRigorousForward`, ``fidelity="rigorous"``) replaces the
T0 skin-depth box-average (:class:`MTForward`) with the exact layered-earth magnetotelluric
impedance (the Wait/Cagniard recursion that is the analytic plane-wave limit of empymod's
layered TE Green's function). These tests assert, on a SMALL truth earth with few stations
and periods, that it (1) runs via empymod and emits the SAME native EDI files as the T0 so
ingestion is unchanged, (2) re-reads through the doc-03 EDI adapter, (3) is physically
sensible — a uniform half-space returns ρ_a = ρ and 45° phase, and apparent resistivity
tracks the true column with skin-depth-correct period→depth behaviour (the shallow clay cap
at short periods, the deep conductor at long periods), (4) DIFFERS from the T0 skin-depth
approximation in the expected way, and (5) is selectable via the fidelity-aware registry
without breaking the T0 forward.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from geosim.ingestion.adapters.mt import MtEdiAdapter
from geosim.ingestion.base import RawSource
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
    MTForward,
    MTRigorousForward,
    get_forward,
)
from geosim.synthgen.forward.em_mt import (
    _MU0,
    _empymod_halfspace_check,
    _layer_model,
    layered_mt_impedance,
)

# --------------------------------------------------------------- a tiny truth earth


def _tiny_scene(seed: int = 4) -> SceneSpec:
    """A SMALL geothermal scene: shallow clay cap (conductor) + deep reservoir conductor."""
    return SceneSpec(
        id="tiny-t1-mt-v1",
        seed=seed,
        frame=FrameSpec(
            xmin=-500, xmax=500, ymin=-500, ymax=500,
            zmin=-1500, zmax=200, dx=160, dy=160, dz=150,
        ),
        surface=SurfaceSpec(kind="tilted-block", base_elev=120.0, tilt_x=0.05),
        layers=(
            LayerSpec("alluvium", "surface", (50.0, 80.0)),
            LayerSpec("volcanics", "conformable", (150.0, 250.0)),
            LayerSpec("basement_granite", "conformable", "fill"),
        ),
        faults=(
            FaultSpec("range-front", trace=((-500, -100), (500, 200)),
                      kind="normal", dip=60, dip_azimuth=90, throw=180, is_conduit=True),
        ),
        geotherm=GeothermSpec(surface_temp=15.0, gradient=45.0),
        anomalies=(
            AnomalySpec(
                "upflow", footprint_center=(0.0, 0.0), footprint_radius_xy=300.0,
                top_elev=120.0, bottom_elev=-1200.0, controlled_by="range-front",
                temp_peak=220.0, alteration_frac=0.9, porosity_boost=0.05,
                salinity_tds=8000.0, fracture_density=0.5,
                clay_cap_top_elev=80.0, clay_cap_thickness=150.0,
            ),
        ),
        rock_physics="default-v1",
    )


@pytest.fixture(scope="module")
def earth():
    return compile_scene(_tiny_scene())


@pytest.fixture()
def acq(tmp_path):
    # few stations (mt_n_periods drives the station grid side) + few periods → seconds.
    return Acquisition(
        mt_n_periods=9, mt_periods=(1.0e-2, 1.0e3), params={"out_dir": str(tmp_path)}
    )


def _rng():
    return np.random.default_rng(123)


# --------------------------------------------------------------- registry wiring


def test_rigorous_forward_is_registered_and_conforms():
    fwd = get_forward("mt", None, fidelity="rigorous")
    assert isinstance(fwd, MTRigorousForward)
    assert isinstance(fwd, ForwardModel)
    assert fwd.fidelity == "rigorous"
    assert fwd.method == "mt" and fwd.submethod is None
    assert ("mt", None) in RIGOROUS_FORWARD_MODELS


def test_plausible_default_unchanged():
    # default fidelity is still T0, and the rigorous selector does not swap it silently.
    assert isinstance(get_forward("mt"), MTForward)
    assert isinstance(get_forward("mt", fidelity="plausible"), MTForward)
    assert not isinstance(get_forward("mt"), MTRigorousForward)


# --------------------------------------------------------------- exact-physics unit checks


def test_halfspace_impedance_is_exact_and_empymod_consistent():
    """A uniform half-space: ρ_a = ρ and phase = 45° at every period (doc 05 §4 row 6).

    This is the analytic plane-wave limit empymod's layered kernel reduces to; the empymod
    cross-check returns the identical ρ_a, proving the rigorous tier is empymod-anchored.
    """
    periods = np.array([1.0e-2, 1.0, 1.0e2])
    z = layered_mt_impedance(np.array([100.0]), np.zeros(0), periods)
    omega = 2.0 * np.pi / periods
    rho_a = np.abs(z) ** 2 / (omega * _MU0)
    phase = np.degrees(np.angle(z))
    np.testing.assert_allclose(rho_a, 100.0, rtol=1e-9)
    np.testing.assert_allclose(phase, 45.0, atol=1e-9)
    # empymod's own μ₀ / convention gives the same half-space ρ_a.
    assert _empymod_halfspace_check(100.0, 1.0) == pytest.approx(100.0, rel=1e-9)


def test_skin_depth_controls_period_to_depth():
    """Short periods see the shallow clay cap, long periods the deep conductor (doc 05 §4.2).

    A conductor(5)/host(500)/conductor(2) column: ρ_a starts near the cap value, rises into
    the resistive host as the skin depth deepens, then falls toward the deep conductor —
    the skin-depth-correct DOI the T0 box-average only approximates.
    """
    depth = np.array([0.0, 50.0, 100.0, 300.0, 600.0, 900.0, 1500.0])
    res = np.array([5.0, 5.0, 500.0, 500.0, 500.0, 2.0, 2.0])
    r, t = _layer_model(depth, res)
    periods = np.array([1.0e-2, 1.0, 1.0e3])
    z = layered_mt_impedance(r, t, periods)
    omega = 2.0 * np.pi / periods
    rho_a = np.abs(z) ** 2 / (omega * _MU0)
    # short T → near the shallow cap; mid T → biased upward by the resistor;
    # long T → diffuses down toward the deep conductor.
    assert rho_a[0] < 30.0                 # shallow conductor dominates
    assert rho_a[1] > rho_a[0]             # resistive host raises mid-band ρ_a
    assert rho_a[2] < rho_a[1]             # deep conductor pulls the long period back down
    assert rho_a[2] < 5.0                  # long period reaches the deep conductor


# --------------------------------------------------------------- native I/O round-trip


def _read_edi(art: Artifact):
    src = RawSource(data=art.path.read_bytes(), filename=art.path.name, crs_hint=None)
    return MtEdiAdapter().parse(src)


def test_rigorous_emits_same_edi_files_and_re_reads(earth, acq):
    """T1 emits one EDI per station (same as T0) that the doc-03 adapter re-reads."""
    arts = MTRigorousForward().simulate(earth, acq, _rng())
    assert len(arts) >= 1
    assert all(isinstance(a, Artifact) for a in arts)
    assert {a.fmt for a in arts} == {"edi"}
    assert all(a.path.suffix == ".edi" for a in arts)

    res = _read_edi(arts[0])
    obs = res.observations[0]
    assert obs.values["resistivity"].size > 0
    assert obs.geometry_kind == "tensor"
    rho = obs.values["resistivity"]
    assert rho.size == acq.mt_n_periods
    assert np.all(np.isfinite(rho)) and np.all(rho > 0)
    assert "phase_mrad" in obs.values  # phase block present and aligned
    assert obs.meta["frequency_hz"] is not None

    prov = arts[0].provenance.to_dict()
    assert prov["source"] == "synthgen"
    assert prov["fidelity"] == "rigorous"
    assert prov["sceneId"] == earth.spec.id
    assert prov["engine"] == "empymod"


def test_rigorous_is_deterministic(earth, tmp_path):
    base = Acquisition(mt_n_periods=9, mt_periods=(1.0e-2, 1.0e3))
    a = MTRigorousForward().simulate(
        earth, replace(base, params={"out_dir": str(tmp_path / "a")}),
        np.random.default_rng(7),
    )
    b = MTRigorousForward().simulate(
        earth, replace(base, params={"out_dir": str(tmp_path / "b")}),
        np.random.default_rng(7),
    )
    ra = _read_edi(a[0]).observations[0].values["resistivity"]
    rb = _read_edi(b[0]).observations[0].values["resistivity"]
    np.testing.assert_allclose(ra, rb)


# --------------------------------------------------------------- tracks the true column


def test_apparent_resistivity_tracks_true_column(earth, acq):
    """The clean (noise-free) rigorous ρ_a equals |Z|²/(ωμ₀) of the station's true column.

    Recomputes the exact impedance from the truth resistivity column under a station and
    checks the emitted (noisy) EDI ρ_a stays within the 4 % measurement noise of it — i.e.
    the curve is the *physics of that column*, skin-depth-correct, not a free parameter.
    """
    from geosim.synthgen.forward.em_mt import _resistivity_column, _station_grid_xy

    arts = MTRigorousForward().simulate(earth, acq, _rng())
    stations = _station_grid_xy(earth, acq.mt_n_periods, spacing=False)
    periods = np.logspace(
        np.log10(acq.mt_periods[0]), np.log10(acq.mt_periods[1]), acq.mt_n_periods
    )
    omega = 2.0 * np.pi / periods

    # match the first artifact's station (ST000 → stations[0]).
    depth, rho = _resistivity_column(earth, stations[0])
    r, t = _layer_model(depth, rho)
    z = layered_mt_impedance(r, t, periods)
    clean = np.abs(z) ** 2 / (omega * _MU0)

    obs = _read_edi(arts[0]).observations[0]
    # EDI writes high→low frequency = long→short period reversed; align by sorting both on
    # period (frequency = 1/period). The adapter keeps the EDI (descending-freq) order.
    freq = np.asarray(obs.meta["frequency_hz"])
    edi_periods = 1.0 / freq
    order = np.argsort(edi_periods)
    got = obs.values["resistivity"][order]
    clean_sorted = clean[np.argsort(periods)]
    # within a few sigma of the 4 % multiplicative noise (allow generous band).
    np.testing.assert_allclose(got, clean_sorted, rtol=0.25)


# --------------------------------------------------------------- T1 vs T0 difference


def test_rigorous_differs_from_t0_and_t0_still_works(earth, tmp_path):
    """T1 (exact impedance) ≠ T0 (skin-depth box-average); both forwards still run."""
    base = Acquisition(mt_n_periods=9, mt_periods=(1.0e-2, 1.0e3))
    acq0 = replace(base, params={"out_dir": str(tmp_path / "t0")})
    acq1 = replace(base, params={"out_dir": str(tmp_path / "t1")})

    a0 = MTForward().simulate(earth, acq0, _rng())        # T0 still works end-to-end
    a1 = MTRigorousForward().simulate(earth, acq1, _rng())
    assert len(a0) == len(a1)

    r0 = _read_edi(a0[0]).observations[0].values["resistivity"]
    r1 = _read_edi(a1[0]).observations[0].values["resistivity"]
    assert r0.size == r1.size

    # the two methods must give a measurable difference (the box-average smears the
    # period→depth structure the exact impedance resolves)...
    rel = np.abs(np.log(r1) - np.log(r0))
    assert np.max(rel) > 0.1
    # ...yet stay the same order of magnitude (both are the same MT sounding).
    assert np.median(r1) == pytest.approx(np.median(r0), rel=5.0)


def test_rigorous_resolves_sharper_dynamic_range_than_t0(earth, tmp_path):
    """The exact impedance resolves a wider ρ_a dynamic range than the smoothed T0.

    The T0 log-skin-depth average blurs the cap→host→reservoir contrast; the rigorous
    layered impedance keeps the sharp shallow-conductor / deep-conductor split, so its
    apparent-resistivity curve spans a larger max/min ratio (doc 05 §4.2 depth split).
    """
    base = Acquisition(mt_n_periods=12, mt_periods=(1.0e-2, 1.0e3))
    a0 = MTForward().simulate(
        earth, replace(base, params={"out_dir": str(tmp_path / "t0")}), _rng()
    )
    a1 = MTRigorousForward().simulate(
        earth, replace(base, params={"out_dir": str(tmp_path / "t1")}), _rng()
    )
    r0 = _read_edi(a0[0]).observations[0].values["resistivity"]
    r1 = _read_edi(a1[0]).observations[0].values["resistivity"]
    span0 = r0.max() / r0.min()
    span1 = r1.max() / r1.min()
    assert span1 > span0
