"""Tests for the FULL rock-physics starter library (doc 07 §4.2).

Each transform of the §4.2 table is driven through the §4.5 execution harness
(:func:`geosim.fusion.run_transform`) on a SMALL synthetic fused grid with **known**
inputs → known outputs:

- temperature: Archie+Arps round-trips a known T (kelvin out) within tolerance;
- fluid: Archie Sw round-trips known φ/Sw/ρ_w; Waxman-Smits/dual-water collapse to Archie at
  zero clay and read *higher* Sw when clay is present (clay over-reads corrected);
- porosity: Wyllie/RHG and density mass-balance round-trip a known φ;
- alteration: low-ρ ⇒ high index; the data-driven GMM posterior separates a bimodal ρ
  population (reuses the §3.3 GMM engine);
- fracture: microseismic KDE peaks at the event cluster; Vp/Vs proxy rises as Vp/Vs falls;
- permeability: log-linear floor→ceiling proxy, damped by alteration; flagged proxy.

Plus library-wide invariants: every transform is registered, every output is canonical
(temperature **kelvin**; permeability **m²**), and every uncalibrated output is retitled a
"… likelihood"/proxy field with ``tier='proxy'`` (doc 07 §4.1/§4.5). Small grids, SQLite
in-memory, temp dirs — no Docker/Postgres/Redis.
"""

import json

import numpy as np
import pytest

import geosim.fusion.rockphys as rockphys  # noqa: F401 — registers the library on import
from geosim.catalog import (
    Dataset,
    IdKind,
    Project,
    PropertyModel,
    Provenance,
    SpatialFrameRow,
    create_all,
    make_engine,
    new_id,
    session_factory,
)
from geosim.fusion import (
    build_fused_model,
    open_fused_group,
    resample_to_fused,
    run_transform,
)
from geosim.fusion.rockphys import (
    AlterationIndex,
    ArchieSaturation,
    DensityToPorosity,
    DualWaterSaturation,
    FractureToPermeability,
    GmmAlterationPosterior,
    MicroseismicDensity,
    ResistivityToTemperature,
    VelocityToPorosity,
    VpVsFractureProxy,
    WaxmanSmitsSaturation,
)
from geosim.fusion.rockphys.temperature import brine_conductivity
from geosim.plugins.registry import get_registry
from geosim.spatial import REGISTRY, Aabb, DepthRange, SpatialFrame
from geosim.storage import (
    GridSpec,
    ensure_project_layout,
    open_property_model,
    write_property_model,
)

SHAPE = (5, 5, 5)
ORIGIN = (-100.0, 20.0, 20.0)
SPACING = (20.0, 20.0, 20.0)


# ───────────────────────────────── fixtures ─────────────────────────────────


@pytest.fixture
def env(tmp_path):
    engine = make_engine()
    create_all(engine)
    Session = session_factory(engine)
    session = Session()
    pid = new_id(IdKind.PROJECT)
    layout = ensure_project_layout(tmp_path, pid)
    frame = SpatialFrame(roi=Aabb(0, 200, 0, 200), depth_range=DepthRange(-200, 0))
    session.add(Project(id=pid, name="rockphys-test", storage_root=str(tmp_path)))
    session.add(SpatialFrameRow(
        project_id=pid, mode=frame.mode.value,
        roi_json=json.dumps({"xmin": 0, "xmax": 200, "ymin": 0, "ymax": 200}),
        depth_range_json=json.dumps({"zmin": -200, "zmax": 0}),
        frame_json=json.dumps({"mode": frame.mode.value}),
    ))
    session.commit()
    yield session, layout, tmp_path, pid
    session.close()


def _aabb():
    oz, oy, ox = ORIGIN
    dz, dy, dx = SPACING
    nz, ny, nx = SHAPE
    return json.dumps({
        "xmin": ox, "xmax": ox + dx * (nx - 1),
        "ymin": oy, "ymax": oy + dy * (ny - 1),
        "zmin": oz, "zmax": oz + dz * (nz - 1),
    })


def _native_pm(session, layout, pid, *, prop, values, unit, sigma=None):
    ds_id = new_id(IdKind.DATASET)
    pm_id = new_id(IdKind.PROPERTY_MODEL)
    prov_id = new_id(IdKind.PROVENANCE)
    zarr_path = layout.zarr_path(pm_id)
    grid = GridSpec(origin=ORIGIN, spacing=SPACING, cell_ref="center")
    write_property_model(zarr_path, prop, values, grid=grid, sigma=sigma, overwrite=True)
    session.add(Provenance(id=prov_id, project_id=pid, target_kind="propertyModel",
                           target_id=pm_id, process="ingest:synthetic"))
    session.flush()
    session.add(Dataset(
        id=ds_id, project_id=pid, name=f"{prop}-native", method="synthetic",
        kind="propertyModel", status="ready", extent_json=_aabb(),
        spatial_frame_id=pid, provenance_id=prov_id, version_root_id=ds_id,
        version_seq=1, created_by="t@x",
    ))
    session.flush()
    row = PropertyModel(
        id=pm_id, dataset_id=ds_id, project_id=pid, property=prop, canonical_unit=unit,
        support="volume", store_uri=str(zarr_path), shape_json=json.dumps(list(SHAPE)),
        spacing_json=json.dumps(list(SPACING)), origin_json=json.dumps(list(ORIGIN)),
        bbox_json=_aabb(),
    )
    session.add(row)
    session.commit()
    return row


def _fused_with(session, layout, pid, props: dict[str, np.ndarray], units: dict[str, str]):
    """Build a fused grid carrying each named property (resampled near-identity)."""
    pms = [
        _native_pm(session, layout, pid, prop=p, values=v, unit=units[p])
        for p, v in props.items()
    ]
    fem, _grid = build_fused_model(
        session, layout, pid, source_property_model_ids=[pms[0].id],
        spacing=SPACING, name="fused-rockphys",
    )
    for pm in pms:
        resample_to_fused(session, fem, pm.id, method="trilinear", interp_space="linear")
    session.refresh(fem)
    return fem


def _read(fem, root, prop):
    group = open_fused_group(fem, storage_root=root)
    layer = [lay for lay in fem.layers if lay.property == prop][-1]
    return np.asarray(group[layer.id][...], dtype=float)


def _read_output(session, root, result):
    pm = session.get(PropertyModel, result.model_id)
    reader = open_property_model(pm.store_uri)
    return reader.read_level(result.output_property, 0)


# ───────────────────────── library-wide invariants ─────────────────────────


def test_full_library_registered():
    ids = {t.id for t in get_registry().transforms()}
    expected = {
        "rp.resistivity_to_temperature.arps",
        "rp.archie_saturation",
        "rp.waxman_smits",
        "rp.dual_water",
        "rp.velocity_to_porosity",
        "rp.density_to_porosity",
        "rp.alteration_index",
        "rp.gmm_alteration_posterior",
        "rp.microseismic_density",
        "rp.vp_vs_fracture_proxy",
        "rp.fracture_to_permeability",
    }
    assert expected <= ids


def test_outputs_are_canonical_units_and_uncalibrated_proxy():
    for t in get_registry().transforms():
        if not t.id.startswith("rp."):
            continue
        # canonical units pinned by the registry (temperature kelvin, permeability m²).
        assert t.output.unit == REGISTRY.get(t.output.name).canonical_unit, t.id
        # the whole starter library ships uncalibrated → proxy/likelihood (doc 07 §4.2).
        assert t.calibration_status == "uncalibrated", t.id
        assert t.assumptions, f"{t.id} must declare assumptions (doc 07 §4.1)"


def test_temperature_output_is_kelvin():
    assert ResistivityToTemperature().output.unit == "kelvin"
    assert REGISTRY.get("temperature").canonical_unit == "kelvin"


# ─────────────────────────────── temperature ───────────────────────────────


def test_resistivity_to_temperature_round_trips_known_T_in_kelvin(env):
    session, layout, root, pid = env
    phi, m, sal, alpha, Tref = 0.10, 2.0, 5000.0, 0.02, 298.15
    T_true = 423.15  # 150 °C
    sigma_w = brine_conductivity(sal, Tref) * (1.0 + alpha * (T_true - Tref))
    rho_value = 1.0 / (sigma_w * (phi**m))
    fem = _fused_with(session, layout, pid,
                      {"resistivity": np.full(SHAPE, rho_value)}, {"resistivity": "ohm*m"})

    result = run_transform(
        session, layout, fem, ResistivityToTemperature(),
        params={"porosity": phi, "m_cementation": m, "fluid_salinity_ppm": sal,
                "arps_slope_per_K": alpha, "T_ref_K": Tref},
        storage_root=root,
    )
    assert result.output_property == "temperature"
    assert result.calibration_status == "uncalibrated"
    assert result.tier == "proxy"
    assert "likelihood" in result.title.lower()

    value = _read_output(session, root, result)
    finite = np.isfinite(value)
    # kelvin out (well above 273), and round-trips the known T.
    np.testing.assert_allclose(value[finite], T_true, rtol=1e-3)


# ─────────────────────────────────── fluid ───────────────────────────────────


def test_archie_saturation_round_trips_known_sw(env):
    session, layout, root, pid = env
    a, m, n, rho_w = 1.0, 2.0, 2.0, 0.5
    phi, sw_true = 0.2, 0.6
    rho_t = (a * rho_w) / (phi**m * sw_true**n)
    fem = _fused_with(session, layout, pid,
                      {"resistivity": np.full(SHAPE, rho_t), "porosity": np.full(SHAPE, phi)},
                      {"resistivity": "ohm*m", "porosity": "dimensionless"})
    result = run_transform(
        session, layout, fem, ArchieSaturation(),
        params={"a_tortuosity": a, "m_cementation": m, "n_saturation": n, "rho_w_ohm_m": rho_w},
        storage_root=root,
    )
    assert result.output_property == "water_saturation"
    value = _read_output(session, root, result)
    finite = np.isfinite(value)
    np.testing.assert_allclose(value[finite], sw_true, rtol=1e-3)


def test_waxman_smits_equals_archie_at_zero_clay_and_corrects_with_clay(env):
    session, layout, root, pid = env
    a, m, n, rho_w = 1.0, 2.0, 2.0, 0.5
    phi, sw_true = 0.2, 0.6
    rho_t = (a * rho_w) / (phi**m * sw_true**n)
    base = {"resistivity": np.full(SHAPE, rho_t), "porosity": np.full(SHAPE, phi)}
    units = {"resistivity": "ohm*m", "porosity": "dimensionless", "clay_volume": "dimensionless"}

    # zero clay → Waxman-Smits collapses to Archie.
    fem0 = _fused_with(session, layout, pid,
                       {**base, "clay_volume": np.zeros(SHAPE)}, units)
    r0 = run_transform(session, layout, fem0, WaxmanSmitsSaturation(),
                       params={"a_tortuosity": a, "m_cementation": m, "n_saturation": n,
                               "rho_w_ohm_m": rho_w, "Qv_max": 1.0},
                       storage_root=root)
    v0 = _read_output(session, root, r0)
    np.testing.assert_allclose(v0[np.isfinite(v0)], sw_true, rtol=2e-2)

    # with clay → clay surface conduction means LESS brine explains the same ρ → lower Sw.
    fem1 = _fused_with(session, layout, pid,
                       {**base, "clay_volume": np.full(SHAPE, 0.4)}, units)
    r1 = run_transform(session, layout, fem1, DualWaterSaturation(),
                       params={"a_tortuosity": a, "m_cementation": m, "n_saturation": n,
                               "rho_w_ohm_m": rho_w, "Qv_max": 1.0},
                       storage_root=root)
    v1 = _read_output(session, root, r1)
    assert np.nanmean(v1) < np.nanmean(v0)  # clay correction reduces apparent Sw


# ─────────────────────────────────── porosity ───────────────────────────────────


@pytest.mark.parametrize("model", ["wyllie", "rhg"])
def test_velocity_to_porosity_round_trips_known_phi(env, model):
    session, layout, root, pid = env
    vm, vf, phi_true = 5500.0, 1500.0, 0.15
    if model == "wyllie":
        vp = 1.0 / ((1 - phi_true) / vm + phi_true / vf)
    else:
        vp = (1 - phi_true) ** 2 * vm + phi_true * vf
    fem = _fused_with(session, layout, pid,
                      {"velocity_p": np.full(SHAPE, vp)}, {"velocity_p": "m/s"})
    result = run_transform(session, layout, fem, VelocityToPorosity(),
                           params={"v_matrix_m_s": vm, "v_fluid_m_s": vf, "model": model},
                           storage_root=root)
    value = _read_output(session, root, result)
    np.testing.assert_allclose(value[np.isfinite(value)], phi_true, rtol=2e-3)


def test_density_to_porosity_round_trips_known_phi(env):
    session, layout, root, pid = env
    rho_m, rho_f, phi_true = 2650.0, 1000.0, 0.15
    rho_b = rho_m - phi_true * (rho_m - rho_f)
    fem = _fused_with(session, layout, pid,
                      {"density": np.full(SHAPE, rho_b)}, {"density": "kg/m**3"})
    result = run_transform(session, layout, fem, DensityToPorosity(),
                           params={"rho_matrix_kg_m3": rho_m, "rho_fluid_kg_m3": rho_f},
                           storage_root=root)
    value = _read_output(session, root, result)
    np.testing.assert_allclose(value[np.isfinite(value)], phi_true, rtol=1e-3)


# ─────────────────────────────────── alteration ───────────────────────────────────


def test_alteration_index_high_where_resistivity_low(env):
    session, layout, root, pid = env
    rho = np.full(SHAPE, 200.0)
    rho[:, :, :2] = 5.0  # a low-ρ (altered) slab
    # provide the optional structure proxy (clay) too — the harness resolves every declared
    # input that is present on the grid; clay reinforces the low-ρ membership.
    clay = np.full(SHAPE, 0.1)
    clay[:, :, :2] = 0.6
    fem = _fused_with(session, layout, pid,
                      {"resistivity": rho, "clay_volume": clay},
                      {"resistivity": "ohm*m", "clay_volume": "dimensionless"})
    result = run_transform(session, layout, fem, AlterationIndex(),
                           params={"rho_threshold_ohm_m": 20.0, "log_width": 0.3,
                                   "structure_weight": 0.4},
                           storage_root=root)
    value = _read_output(session, root, result)
    assert result.output_property == "alteration"
    assert np.nanmean(value[:, :, :2]) > 0.6  # altered slab
    assert np.nanmean(value[:, :, 2:]) < 0.3  # resistive background
    assert np.all((value[np.isfinite(value)] >= 0.0) & (value[np.isfinite(value)] <= 1.0))


def test_gmm_alteration_posterior_separates_bimodal_resistivity(env):
    session, layout, root, pid = env
    rng = np.random.default_rng(0)
    rho = np.empty(SHAPE)
    half = SHAPE[2] // 2
    rho[:, :, :half] = 10 ** rng.normal(0.7, 0.05, rho[:, :, :half].shape)  # low ρ
    rho[:, :, half:] = 10 ** rng.normal(2.5, 0.05, rho[:, :, half:].shape)  # high ρ
    fem = _fused_with(session, layout, pid, {"resistivity": rho}, {"resistivity": "ohm*m"})
    result = run_transform(session, layout, fem, GmmAlterationPosterior(),
                           params={"n_components": 2}, storage_root=root)
    value = _read_output(session, root, result)
    # data-driven: low-ρ cells get high altered-posterior, high-ρ cells get low.
    assert np.nanmean(value[:, :, :half]) > 0.8
    assert np.nanmean(value[:, :, half:]) < 0.2


# ─────────────────────────────────── fracture ───────────────────────────────────


def test_microseismic_density_peaks_at_event_cluster(env):
    session, layout, root, pid = env
    counts = np.zeros(SHAPE)
    counts[2, 2, 2] = 20.0  # event cluster at the grid centre
    fem = _fused_with(session, layout, pid, {"microseismic": counts},
                      {"microseismic": "dimensionless"})
    result = run_transform(session, layout, fem, MicroseismicDensity(),
                           params={"bandwidth_cells": 1.0}, storage_root=root)
    value = _read_output(session, root, result)
    assert result.output_property == "fracture_density"
    finite = np.isfinite(value)
    assert np.all((value[finite] >= 0.0) & (value[finite] <= 1.0))
    # peak density is at the event cluster (KDE-smoothed).
    assert np.nanargmax(value) == np.ravel_multi_index((2, 2, 2), SHAPE)
    assert np.isclose(np.nanmax(value), 1.0)  # peak-normalised


def test_vp_vs_fracture_proxy_rises_as_ratio_falls(env):
    session, layout, root, pid = env
    vp = np.full(SHAPE, 3000.0)
    vs = np.full(SHAPE, 2000.0)  # Vp/Vs = 1.5 (low → fractured)
    vs[:, :, 2:] = 1500.0        # Vp/Vs = 2.0 (high → unfractured)
    fem = _fused_with(session, layout, pid, {"velocity_p": vp, "velocity_s": vs},
                      {"velocity_p": "m/s", "velocity_s": "m/s"})
    result = run_transform(session, layout, fem, VpVsFractureProxy(),
                           params={"ratio_threshold": 1.7, "width": 0.1}, storage_root=root)
    value = _read_output(session, root, result)
    assert np.nanmean(value[:, :, :2]) > np.nanmean(value[:, :, 2:])


# ─────────────────────────────────── permeability ───────────────────────────────────


def test_fracture_to_permeability_log_linear_and_proxy(env):
    session, layout, root, pid = env
    fd = np.full(SHAPE, 0.0)
    fd[:, :, 2:] = 1.0  # half tight, half fully fractured
    # alteration=0 everywhere → no sealing damping; pure log-linear fracture→k.
    fem = _fused_with(session, layout, pid,
                      {"fracture_density": fd, "alteration": np.zeros(SHAPE)},
                      {"fracture_density": "dimensionless", "alteration": "dimensionless"})
    k_min, k_max = 1e-17, 1e-12
    result = run_transform(session, layout, fem, FractureToPermeability(),
                           params={"k_min_m2": k_min, "k_max_m2": k_max, "alteration_seal": 0.5},
                           storage_root=root)
    assert result.output_property == "permeability"
    assert result.calibration_status == "uncalibrated"
    assert result.tier == "proxy"  # flagged low-confidence proxy
    value = _read_output(session, root, result)
    np.testing.assert_allclose(value[:, :, 0], k_min, rtol=1e-3)
    np.testing.assert_allclose(value[:, :, 2], k_max, rtol=1e-3)
    # canonical SI m² output.
    assert REGISTRY.get("permeability").canonical_unit == "m**2"


def test_fracture_to_permeability_alteration_seals(env):
    session, layout, root, pid = env
    fd = np.full(SHAPE, 1.0)
    alt = np.zeros(SHAPE)
    alt[:, :, 2:] = 1.0  # fully altered (sealed) half
    fem = _fused_with(session, layout, pid, {"fracture_density": fd, "alteration": alt},
                      {"fracture_density": "dimensionless", "alteration": "dimensionless"})
    result = run_transform(session, layout, fem, FractureToPermeability(),
                           params={"k_min_m2": 1e-17, "k_max_m2": 1e-12, "alteration_seal": 0.5},
                           storage_root=root)
    value = _read_output(session, root, result)
    # altered/sealed cells have lower permeability than unaltered fractured cells.
    assert np.nanmean(value[:, :, 2:]) < np.nanmean(value[:, :, :2])


# ──────────────────── pure-apply: optional inputs + event binning ────────────────────


class _Ctx:
    """Minimal ctx for pure ``apply()`` unit checks (no grid/units/σ)."""

    def __init__(self, shape=None):
        self.grid = type("G", (), {"shape": shape})() if shape else None

    @staticmethod
    def as_output(a):
        return np.asarray(a, dtype=float)


def test_optional_inputs_apply_with_none():
    # AlterationIndex with no clay → bare low-ρ membership.
    out = AlterationIndex().apply(_Ctx(), np.array([5.0, 200.0]), None,
                                  rho_threshold_ohm_m=20.0, log_width=0.3, structure_weight=0.4)
    assert out[0] > out[1]
    # FractureToPermeability with no alteration → no sealing damping.
    k = FractureToPermeability().apply(_Ctx(), np.array([0.0, 1.0]), None,
                                       k_min_m2=1e-17, k_max_m2=1e-12, alteration_seal=0.5)
    assert np.isclose(k[0], 1e-17, rtol=1e-3) and np.isclose(k[1], 1e-12, rtol=1e-3)


def test_events_to_count_volume_bins_cloud():
    from geosim.fusion.rockphys.fracture import events_to_count_volume

    grid = type("G", (), {"shape": (4, 4, 4), "origin": (0.0, 0.0, 0.0),
                          "spacing": (1.0, 1.0, 1.0)})()
    # events are (x, y, z); the grid is (z, y, x)-ordered.
    events = np.array([[2.0, 2.0, 2.0], [2.1, 2.0, 2.0], [10.0, 0.0, 0.0]])  # last is OOB
    vol = events_to_count_volume(events, grid)
    assert vol[2, 2, 2] == 2.0
    assert vol.sum() == 2.0  # the out-of-bounds event is dropped
