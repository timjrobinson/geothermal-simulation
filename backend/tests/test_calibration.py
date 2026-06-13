"""Tests for the calibration workflow (doc 07 §4.8).

Small/coarse grids, local temp dirs, SQLite in-memory — no Docker/Postgres/Redis.

The calibration loop is the CENTRE of the rock-physics workflow (doc 07 §4.8): a transform is
born uncalibrated (proxy/likelihood) and ground-truth well probes promote it. These tests:

- synthesize a transform with a KNOWN true param, generate well-probe "measurements" from the
  truth (the transform evaluated at the true param along a well path), fit → recover the param
  within tolerance **with a σ** (a parameter distribution, not a point fit, §4.8 ②);
- check the re-run promotes ``calibration_status`` → ``well_calibrated`` and tier proxy →
  quantitative NEAR the well, and that cells FAR from any probe stay proxy / "likelihood"
  (spatially honest promotion, §4.8 ④);
- check the fitted param σ is written onto the re-run's params so §5.2 propagation carries it;
- check ``probes_from_deviation_survey`` lands a vertical well on the right column;
- check synthetic-only truth scoring is flagged ``synthetic_only`` and recovers a tight RMSE;
- exercise ``POST /fused/{gridId}/calibrate`` (sync).
"""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from geosim.api.app import Settings, create_app
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
    InputSpec,
    OutputSpec,
    Param,
    Probe,
    Transform,
    build_fused_model,
    calibrate_transform,
    fit_transform_params,
    fused_grid_from_row,
    probes_from_deviation_survey,
    promote_spatial,
    resample_to_fused,
    run_transform,
    score_against_truth,
)
from geosim.fusion.grid import open_fused_group
from geosim.plugins import register
from geosim.spatial import Aabb, DepthRange, SpatialFrame
from geosim.storage import (
    GridSpec,
    ensure_project_layout,
    open_property_model,
    write_property_model,
)

# ───────────────────────────── toy transform ─────────────────────────────


class LinearTempFromResistivity(Transform):
    """``temperature_K = a·resistivity + b``; ``a`` is the calibratable site param.

    Born uncalibrated → output is a proxy/likelihood field; a well-probe run recovers ``a``
    and promotes it. Linear so the fit is exact and the σ check is clean.
    """

    id = "test.calib_linear_temp"
    version = "1.0.0"
    title = "Linear temperature (calib test)"
    target = "temperature"
    inputs = [InputSpec("resistivity", unit="ohm*m", required=True)]
    output = OutputSpec("temperature", unit="kelvin", valid_range=(0.0, 1e6), colormap="thermal")
    params = [Param("a", float, default=1.0, range=(0.0, 100.0)),
              Param("b", float, default=300.0, range=(0.0, 1000.0))]
    assumptions = ["linear toy relationship"]
    calibration_status = "uncalibrated"

    def apply(self, ctx, resistivity, *, a, b):
        return ctx.as_output(a * resistivity + b)


TRUE_A = 2.5
TRUE_B = 300.0


@pytest.fixture(scope="module", autouse=True)
def _register_transforms():
    register.transform(LinearTempFromResistivity())


# ───────────────────────────────── fixtures ─────────────────────────────────


def _aabb(xmin, xmax, ymin, ymax, zmin, zmax) -> str:
    return json.dumps(
        {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "zmin": zmin, "zmax": zmax}
    )


@pytest.fixture
def env(tmp_path):
    engine = make_engine()
    create_all(engine)
    Session = session_factory(engine)
    session = Session()
    storage_root = tmp_path
    pid = new_id(IdKind.PROJECT)
    layout = ensure_project_layout(storage_root, pid)
    frame = SpatialFrame(roi=Aabb(0, 200, 0, 200), depth_range=DepthRange(-200, 0))
    session.add(Project(id=pid, name="calib-test", storage_root=str(storage_root)))
    session.add(SpatialFrameRow(
        project_id=pid, mode=frame.mode.value,
        roi_json=json.dumps({"xmin": 0, "xmax": 200, "ymin": 0, "ymax": 200}),
        depth_range_json=json.dumps({"zmin": -200, "zmax": 0}),
        frame_json=json.dumps({"mode": frame.mode.value}),
    ))
    session.commit()
    yield session, layout, storage_root, pid
    session.close()


def _make_native_resistivity(session, layout, pid, *, sigma_rel=0.05):
    ds_id = new_id(IdKind.DATASET)
    pm_id = new_id(IdKind.PROPERTY_MODEL)
    prov_id = new_id(IdKind.PROVENANCE)
    shape = (6, 6, 6)
    origin = (-100.0, 20.0, 20.0)
    spacing = (20.0, 20.0, 20.0)
    rng = np.random.default_rng(0)
    res = 50.0 + rng.uniform(0, 50, shape)
    sigma = res * sigma_rel
    zarr_path = layout.zarr_path(pm_id)
    grid = GridSpec(origin=origin, spacing=spacing, cell_ref="center")
    write_property_model(zarr_path, "resistivity", res, grid=grid, sigma=sigma, overwrite=True)
    nz, ny, nx = shape
    oz, oy, ox = origin
    dz, dy, dx = spacing
    bbox = _aabb(ox, ox + dx * (nx - 1), oy, oy + dy * (ny - 1), oz, oz + dz * (nz - 1))
    session.add(Provenance(id=prov_id, project_id=pid, target_kind="propertyModel",
                           target_id=pm_id, process="ingest:synthetic"))
    session.flush()
    session.add(Dataset(
        id=ds_id, project_id=pid, name="resistivity-native", method="mt", kind="propertyModel",
        status="ready", extent_json=bbox, spatial_frame_id=pid, provenance_id=prov_id,
        version_root_id=ds_id, version_seq=1, created_by="t@x",
    ))
    session.flush()
    row = PropertyModel(
        id=pm_id, dataset_id=ds_id, project_id=pid, property="resistivity",
        canonical_unit="ohm*m", support="volume", store_uri=str(zarr_path),
        shape_json=json.dumps([nz, ny, nx]), spacing_json=json.dumps(list(spacing)),
        origin_json=json.dumps(list(origin)), bbox_json=bbox,
    )
    session.add(row)
    session.commit()
    return row, origin, spacing, shape


def _build_fused(session, layout, pid, **kw):
    res_pm, origin, spacing, shape = _make_native_resistivity(session, layout, pid, **kw)
    fem, _grid = build_fused_model(
        session, layout, pid, source_property_model_ids=[res_pm.id],
        spacing=spacing, name="fused-calib",
    )
    resample_to_fused(session, fem, res_pm.id, method="trilinear", interp_space="linear")
    session.refresh(fem)
    return fem, origin, spacing, shape


def _fused_resistivity(fem, root):
    group = open_fused_group(fem, storage_root=root)
    layer = [lay for lay in fem.layers if lay.property == "resistivity"][-1]
    return np.asarray(group[layer.id][...], dtype=float)


def _probes_from_truth(fem, root, columns, *, noise=0.0, seed=1):
    """Build probes by evaluating the TRUE relationship at sampled grid columns.

    ``columns`` is a list of ``(iy, ix)`` grid columns the synthetic well penetrates; we take
    every depth in that column as a probe (measured = a_true*res + b_true [+ noise]).
    """
    grid = fused_grid_from_row(fem)
    res = _fused_resistivity(fem, root)
    z, y, x = grid.axis_coords()
    rng = np.random.default_rng(seed)
    probes = []
    for (iy, ix) in columns:
        for iz in range(grid.shape[0]):
            r = res[iz, iy, ix]
            if not np.isfinite(r):
                continue
            measured = TRUE_A * r + TRUE_B + (rng.normal(0, noise) if noise else 0.0)
            probes.append(Probe(z=float(z[iz]), y=float(y[iy]), x=float(x[ix]),
                                measured=float(measured), unit="kelvin"))
    return probes


# ─────────────────────────── ② fit recovers param + σ ───────────────────────────


def test_fit_recovers_true_param_with_sigma(env):
    session, layout, root, pid = env
    fem, _o, _s, _sh = _build_fused(session, layout, pid)
    probes = _probes_from_truth(fem, root, [(2, 2), (3, 3)])

    fit = fit_transform_params(
        session, fem, LinearTempFromResistivity(), probes, ["a"],
        params={"b": TRUE_B}, storage_root=root,
    )
    assert fit.converged
    assert fit.n_probes == len(probes)
    # recovers a_true within tolerance, with a (finite) σ — a parameter distribution.
    assert fit.params["a"] == pytest.approx(TRUE_A, abs=1e-3)
    assert "a" in fit.param_sigma and np.isfinite(fit.param_sigma["a"])
    assert fit.param_sigma["a"] >= 0.0
    # noiseless ⇒ essentially exact fit.
    assert fit.rms_residual < 1e-3


def test_fit_recovers_param_under_noise_with_larger_sigma(env):
    session, layout, root, pid = env
    fem, _o, _s, _sh = _build_fused(session, layout, pid)
    clean = _probes_from_truth(fem, root, [(2, 2), (3, 3)], noise=0.0)
    noisy = _probes_from_truth(fem, root, [(2, 2), (3, 3)], noise=5.0, seed=7)

    fit_clean = fit_transform_params(session, fem, LinearTempFromResistivity(), clean, ["a"],
                                     params={"b": TRUE_B}, storage_root=root)
    fit_noisy = fit_transform_params(session, fem, LinearTempFromResistivity(), noisy, ["a"],
                                     params={"b": TRUE_B}, storage_root=root)
    # still near the truth, but the fit σ is larger under measurement noise.
    assert fit_noisy.params["a"] == pytest.approx(TRUE_A, abs=0.1)
    assert fit_noisy.param_sigma["a"] > fit_clean.param_sigma["a"]


def test_fit_two_params(env):
    session, layout, root, pid = env
    fem, _o, _s, _sh = _build_fused(session, layout, pid)
    probes = _probes_from_truth(fem, root, [(2, 2), (3, 3), (4, 4)])
    fit = fit_transform_params(session, fem, LinearTempFromResistivity(), probes, ["a", "b"],
                               storage_root=root)
    assert fit.params["a"] == pytest.approx(TRUE_A, abs=1e-2)
    assert fit.params["b"] == pytest.approx(TRUE_B, abs=1.0)


def test_fit_rejects_unknown_param(env):
    session, layout, root, pid = env
    fem, _o, _s, _sh = _build_fused(session, layout, pid)
    probes = _probes_from_truth(fem, root, [(2, 2)])
    with pytest.raises(ValueError, match="not a tunable param"):
        fit_transform_params(session, fem, LinearTempFromResistivity(), probes, ["nope"],
                             storage_root=root)


def test_fit_rejects_all_offgrid_probes(env):
    session, layout, root, pid = env
    fem, _o, _s, _sh = _build_fused(session, layout, pid)
    # probes far outside the grid footprint → all NaN input → no usable pairs.
    far = [Probe(z=1e6, y=1e6, x=1e6, measured=400.0, unit="kelvin")]
    with pytest.raises(ValueError, match="no usable probe pairs"):
        fit_transform_params(session, fem, LinearTempFromResistivity(), far, ["a"],
                             storage_root=root)


# ─────────────────────────── ④ spatially-honest promotion ───────────────────────────


def test_promote_spatial_near_vs_far(env):
    session, layout, root, pid = env
    fem, _o, _s, _sh = _build_fused(session, layout, pid)
    grid = fused_grid_from_row(fem)
    # one probe at a single column centre.
    z, y, x = grid.axis_coords()
    probe = Probe(z=float(z[3]), y=float(y[3]), x=float(x[3]), measured=400.0, unit="kelvin")
    promoted, stats = promote_spatial(grid, [probe], resolving_distance=30.0)
    # the probe's own cell (and immediate neighbours within 30 m) are promoted; far ones aren't.
    assert promoted[3, 3, 3]
    assert not promoted[0, 0, 0]
    assert 0.0 < stats["promoted_fraction"] < 1.0
    assert stats["n_promoted"] == int(promoted.sum())


def test_promote_spatial_no_probes_promotes_nothing(env):
    session, layout, root, pid = env
    fem, _o, _s, _sh = _build_fused(session, layout, pid)
    grid = fused_grid_from_row(fem)
    promoted, stats = promote_spatial(grid, [], resolving_distance=1e9)
    assert not promoted.any()
    assert stats["promoted_fraction"] == 0.0


def test_promote_spatial_rejects_nonpositive_distance(env):
    session, layout, root, pid = env
    fem, _o, _s, _sh = _build_fused(session, layout, pid)
    grid = fused_grid_from_row(fem)
    with pytest.raises(ValueError, match="resolving_distance must be positive"):
        promote_spatial(grid, [], resolving_distance=0.0)


# ─────────────────── full loop: fit → re-run → promote (doc 07 §4.8) ───────────────────


def test_calibrate_promotes_status_and_carries_param_sigma(env):
    session, layout, root, pid = env
    fem, _o, _s, _sh = _build_fused(session, layout, pid)
    probes = _probes_from_truth(fem, root, [(2, 2), (3, 3)], noise=3.0, seed=3)

    result = calibrate_transform(
        session, layout, fem, LinearTempFromResistivity(), probes, ["a"],
        resolving_distance=30.0, params={"b": TRUE_B}, storage_root=root,
    )
    # promotion: status well_calibrated, near cells quantitative, far cells stay proxy.
    assert result.calibration_status == "well_calibrated"
    assert result.near_cell_tier == "quantitative"
    assert result.far_cell_tier == "proxy"
    assert 0.0 < result.promoted_fraction < 1.0
    # the re-run produced a calibrated (quantitative) derived volume, not a proxy likelihood.
    assert result.transform_result.calibration_status == "well_calibrated"
    assert result.transform_result.tier == "quantitative"
    assert "likelihood" not in result.transform_result.title.lower()

    # recovered param near truth.
    assert result.fit.params["a"] == pytest.approx(TRUE_A, abs=0.1)

    # the fitted σ propagated: σ_temperature ≈ |res|·σ_a where σ_a>0 (param term dominant here).
    pm = session.get(PropertyModel, result.transform_result.model_id)
    reader = open_property_model(pm.store_uri)
    sigma_vol = reader.read_sigma_level("temperature", 0)
    res = _fused_resistivity(fem, root)
    finite = np.isfinite(res) & np.isfinite(sigma_vol)
    sig_a = result.fit.param_sigma["a"]
    assert sig_a > 0.0
    # σ has a contribution (res·σ_a) from the calibrated param — strictly positive everywhere.
    assert np.all(sigma_vol[finite] > 0.0)
    # and it scales with resistivity (the param-σ term): corr(σ, res) high.
    assert np.corrcoef(sigma_vol[finite], res[finite])[0, 1] > 0.5

    # provenance records the calibration block (doc 07 §4.3/§4.8).
    prov = session.get(Provenance, session.get(Dataset, pm.dataset_id).provenance_id)
    deriv = json.loads(prov.params_json)["derivation"]
    assert deriv["calibrationStatus"] == "well_calibrated"
    assert deriv["calibration"]["fittedParams"] == ["a"]
    assert deriv["calibration"]["nProbes"] == len(probes)
    assert "promotion" in deriv["calibration"]


def test_calibrate_uncalibrated_baseline_is_proxy(env):
    """Sanity: WITHOUT calibration the same transform is a proxy/likelihood field (§4.1)."""
    session, layout, root, pid = env
    fem, _o, _s, _sh = _build_fused(session, layout, pid)
    base = run_transform(session, layout, fem, LinearTempFromResistivity(),
                         params={"a": 1.0, "b": TRUE_B}, storage_root=root)
    assert base.calibration_status == "uncalibrated"
    assert base.tier == "proxy"
    assert "likelihood" in base.title.lower()


# ─────────────────── well deviation survey → Engineering-XYZ probes ───────────────────


def test_probes_from_vertical_deviation_survey(env):
    session, layout, root, pid = env
    fem, origin, spacing, shape = _build_fused(session, layout, pid)
    grid = fused_grid_from_row(fem)
    z, y, x = grid.axis_coords()
    # A vertical well at (x=x[3], y=y[3]); wellhead elevation at the grid top (z max).
    wellhead = [float(x[3]), float(y[3])]
    kb = float(z.max())
    # deviation survey: vertical (inc=0) from MD 0 to the full depth range.
    total_md = float(z.max() - z.min())
    survey = [[0.0, 0.0, 0.0], [total_md, 0.0, 0.0]]
    # measured log: temperatures sampled every cell of depth.
    mds = np.linspace(0.0, total_md, shape[0])
    vals = np.full(shape[0], 400.0)
    probes = probes_from_deviation_survey(
        survey, wellhead, mds, vals, unit="kelvin", kb_elev=kb,
    )
    assert len(probes) == shape[0]
    # the well is vertical at (x[3], y[3]): every probe shares that x/y.
    for p in probes:
        assert p.x == pytest.approx(float(x[3]), abs=1e-6)
        assert p.y == pytest.approx(float(y[3]), abs=1e-6)
    # z descends from kb (top) down through the column.
    zs = [p.z for p in probes]
    assert zs[0] == pytest.approx(kb, abs=1e-6)
    assert zs[-1] < zs[0]


# ─────────────────── synthetic-only truth scoring (NEVER real data) ───────────────────


def test_score_against_truth_is_synthetic_only_and_tight(env):
    session, layout, root, pid = env
    fem, _o, _s, _sh = _build_fused(session, layout, pid)
    res = _fused_resistivity(fem, root)
    truth = TRUE_A * res + TRUE_B  # synthetic doc-05 truth field
    probes = _probes_from_truth(fem, root, [(2, 2), (3, 3)])

    result = calibrate_transform(
        session, layout, fem, LinearTempFromResistivity(), probes, ["a"],
        resolving_distance=1e9, params={"b": TRUE_B},
        truth_value=truth, storage_root=root,
    )
    assert result.truth_score is not None
    assert result.truth_score["synthetic_only"] is True
    # recovered param ⇒ calibrated volume matches truth tightly.
    assert result.truth_score["rmse"] < 1.0
    assert result.truth_score["n"] > 0


def test_score_against_truth_direct():
    cal = np.array([[300.0, 400.0], [np.nan, 500.0]])
    truth = np.array([[301.0, 399.0], [350.0, 500.0]])
    s = score_against_truth(cal, truth)
    assert s["synthetic_only"] is True
    assert s["n"] == 3  # the NaN cell is dropped
    assert s["rmse"] == pytest.approx(np.sqrt((1 + 1 + 0) / 3), abs=1e-9)


# ─────────────────────────────── REST endpoint ───────────────────────────────


@pytest.fixture
def client_env(tmp_path):
    storage_root = tmp_path / "store"
    storage_root.mkdir()
    app = create_app(Settings(storage_root=str(storage_root)))
    client = TestClient(app)
    Session = app.state.session_factory
    session = Session()
    pid = new_id(IdKind.PROJECT)
    layout = ensure_project_layout(storage_root, pid)
    frame = SpatialFrame(roi=Aabb(0, 200, 0, 200), depth_range=DepthRange(-200, 0))
    session.add(Project(id=pid, name="rest-calib", storage_root=str(storage_root)))
    session.add(SpatialFrameRow(
        project_id=pid, mode=frame.mode.value,
        roi_json=json.dumps({"xmin": 0, "xmax": 200, "ymin": 0, "ymax": 200}),
        depth_range_json=json.dumps({"zmin": -200, "zmax": 0}),
        frame_json=json.dumps({"mode": frame.mode.value}),
    ))
    session.commit()
    fem, _o, _s, _sh = _build_fused(session, layout, pid)
    fem_id = fem.id
    probes = _probes_from_truth(fem, storage_root, [(2, 2), (3, 3)])
    probe_payload = [{"z": p.z, "y": p.y, "x": p.x, "measured": p.measured, "unit": p.unit}
                     for p in probes]
    session.close()
    yield client, pid, fem_id, probe_payload


def test_endpoint_calibrate_sync(client_env):
    client, pid, fem_id, probe_payload = client_env
    r = client.post(f"/fused/{fem_id}/calibrate", json={
        "project_id": pid, "transform_id": "test.calib_linear_temp",
        "fit_params": ["a"], "resolving_distance": 30.0,
        "params": {"b": TRUE_B}, "probes": probe_payload,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "sync"
    assert body["calibration_status"] == "well_calibrated"
    assert body["near_cell_tier"] == "quantitative"
    assert body["far_cell_tier"] == "proxy"
    assert body["fit"]["params"]["a"] == pytest.approx(TRUE_A, abs=0.1)
    assert 0.0 < body["promoted_fraction"] < 1.0


def test_endpoint_calibrate_unknown_transform_404(client_env):
    client, pid, fem_id, probe_payload = client_env
    r = client.post(f"/fused/{fem_id}/calibrate", json={
        "project_id": pid, "transform_id": "nope", "fit_params": ["a"],
        "resolving_distance": 30.0, "probes": probe_payload,
    })
    assert r.status_code == 404, r.text


def test_endpoint_calibrate_missing_probes_400(client_env):
    client, pid, fem_id, _probe_payload = client_env
    r = client.post(f"/fused/{fem_id}/calibrate", json={
        "project_id": pid, "transform_id": "test.calib_linear_temp",
        "fit_params": ["a"], "resolving_distance": 30.0,
    })
    assert r.status_code == 400, r.text
