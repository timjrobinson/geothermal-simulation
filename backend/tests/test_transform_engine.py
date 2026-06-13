"""Tests for the rock-physics transform engine (doc 07 §4–§5).

Small/coarse grids, local temp dirs, SQLite in-memory — no Docker/Postgres/Redis.

Registers a couple of toy transforms (a well-calibrated linear ``y = a·x1 + b`` and an
uncalibrated likelihood transform) and drives them through the §4.5 harness on a small
fused grid, checking:

- a derived PropertyModel + paired σ are written, σ matches the analytic delta-method;
- the NaN coverage mask is honored (a cell missing a required input → NaN out);
- an uncalibrated transform retitles its output ``"… likelihood"`` and stamps tier=proxy;
- Monte-Carlo σ agrees with the delta-method on the linear case;
- ``GET /transforms`` palette + ``POST /fused/{gridId}/transform`` endpoints work
  (sync + job-based / Monte-Carlo paths).
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
    Transform,
    build_fused_model,
    fused_grid_from_row,
    resample_to_fused,
    run_transform,
)
from geosim.plugins import register
from geosim.spatial import Aabb, DepthRange, SpatialFrame
from geosim.storage import (
    GridSpec,
    ensure_project_layout,
    open_property_model,
    write_property_model,
)

# ───────────────────────────── toy transforms ─────────────────────────────


class LinearTempFromResistivity(Transform):
    """Well-calibrated linear toy: ``temperature_K = a·resistivity + b`` (analytic σ check).

    Linear so the delta-method Jacobian is exact and Monte-Carlo must agree. Declares a
    param σ on ``a`` so the parameter-uncertainty term of σ propagation is exercised.
    """

    id = "test.linear_temp"
    version = "1.0.0"
    title = "Linear temperature (test)"
    target = "temperature"
    inputs = [InputSpec("resistivity", unit="ohm*m", required=True)]
    output = OutputSpec("temperature", unit="kelvin", valid_range=(0.0, 1e6), colormap="thermal")
    params = [Param("a", float, default=2.0, range=(0.0, 100.0), sigma=0.0),
              Param("b", float, default=300.0, range=(0.0, 1000.0))]
    assumptions = ["linear toy relationship"]
    calibration_status = "well_calibrated"

    def apply(self, ctx, resistivity, *, a, b):
        return ctx.as_output(a * resistivity + b)


class UncalibratedLikelihood(Transform):
    """Uncalibrated toy → must be retitled '… likelihood' + tier proxy (doc 07 §4.1/§4.5)."""

    id = "test.uncal_temp"
    version = "0.1.0"
    title = "Uncalibrated temperature (test)"
    target = "temperature"
    inputs = [InputSpec("resistivity", unit="ohm*m", required=True)]
    output = OutputSpec("temperature", unit="kelvin", valid_range=(273.0, 673.0),
                        proxy_when_uncalibrated=True)
    params = [Param("scale", float, default=1.0, range=(0.1, 10.0))]
    calibration_status = "uncalibrated"

    def apply(self, ctx, resistivity, *, scale):
        # bounded so it lands inside valid_range without much clamping
        return ctx.as_output(400.0 + scale * np.log10(np.maximum(resistivity, 1.0)) * 10.0)


@pytest.fixture(scope="module", autouse=True)
def _register_transforms():
    register.transform(LinearTempFromResistivity())
    register.transform(UncalibratedLikelihood())


# ───────────────────────────────── fixtures ─────────────────────────────────


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
    session.add(Project(id=pid, name="transform-test", storage_root=str(storage_root)))
    session.add(SpatialFrameRow(
        project_id=pid, mode=frame.mode.value,
        roi_json=json.dumps({"xmin": 0, "xmax": 200, "ymin": 0, "ymax": 200}),
        depth_range_json=json.dumps({"zmin": -200, "zmax": 0}),
        frame_json=json.dumps({"mode": frame.mode.value}),
    ))
    session.commit()
    yield session, layout, storage_root, pid
    session.close()


def _aabb(xmin, xmax, ymin, ymax, zmin, zmax) -> str:
    return json.dumps(
        {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "zmin": zmin, "zmax": zmax}
    )


def _make_native_pm(session, layout, pid, *, prop, values, origin, spacing, unit,
                    method="mt", sigma=None):
    ds_id = new_id(IdKind.DATASET)
    pm_id = new_id(IdKind.PROPERTY_MODEL)
    prov_id = new_id(IdKind.PROVENANCE)
    zarr_path = layout.zarr_path(pm_id)
    grid = GridSpec(origin=origin, spacing=spacing, cell_ref="center")
    write_property_model(zarr_path, prop, values, grid=grid, sigma=sigma, overwrite=True)

    nz, ny, nx = values.shape
    oz, oy, ox = origin
    dz, dy, dx = spacing
    bbox = _aabb(ox, ox + dx * (nx - 1), oy, oy + dy * (ny - 1), oz, oz + dz * (nz - 1))
    session.add(Provenance(id=prov_id, project_id=pid, target_kind="propertyModel",
                           target_id=pm_id, process="ingest:synthetic"))
    session.flush()
    session.add(Dataset(
        id=ds_id, project_id=pid, name=f"{prop}-native", method=method, kind="propertyModel",
        status="ready", extent_json=bbox, spatial_frame_id=pid, provenance_id=prov_id,
        version_root_id=ds_id, version_seq=1, created_by="t@x",
    ))
    session.flush()
    row = PropertyModel(
        id=pm_id, dataset_id=ds_id, project_id=pid, property=prop, canonical_unit=unit,
        support="volume", store_uri=str(zarr_path), shape_json=json.dumps([nz, ny, nx]),
        spacing_json=json.dumps(list(spacing)), origin_json=json.dumps(list(origin)),
        bbox_json=bbox,
    )
    session.add(row)
    session.commit()
    return row


def _build_fused_with_resistivity(session, layout, pid, *, nan_corner=False, sigma_rel=0.1):
    """A small fused grid with a resistivity layer (known σ) resampled in.

    ``nan_corner`` plants a NaN octant in the native model → that region is outside the
    coverage mask and must read NaN through the transform (doc 07 §2.3/§4.5 step 3).
    """
    shape = (6, 6, 6)
    rng = np.random.default_rng(0)
    res = 50.0 + rng.uniform(0, 50, shape)  # 50..100 ohm·m, smoothly varying
    sigma = res * sigma_rel
    if nan_corner:
        res[:3, :3, :3] = np.nan
        sigma[:3, :3, :3] = np.nan
    origin = (-100.0, 20.0, 20.0)
    spacing = (20.0, 20.0, 20.0)
    res_pm = _make_native_pm(
        session, layout, pid, prop="resistivity", values=res, origin=origin,
        spacing=spacing, unit="ohm*m", method="mt", sigma=sigma,
    )
    fem, _grid = build_fused_model(
        session, layout, pid, source_property_model_ids=[res_pm.id],
        spacing=spacing, name="fused-transform",
    )
    # Use the SAME grid as the native model so resampling is a near-identity (no extra
    # interpolation σ inflation) — keeps the analytic σ check clean.
    resample_to_fused(session, fem, res_pm.id, method="trilinear", interp_space="linear")
    session.refresh(fem)
    return fem, res_pm


# ─────────────────────────── harness: derived PM + σ ───────────────────────────


def test_linear_transform_writes_derived_pm_and_sigma(env):
    session, layout, root, pid = env
    fem, _res = _build_fused_with_resistivity(session, layout, pid)

    result = run_transform(
        session, layout, fem, LinearTempFromResistivity(),
        params={"a": 2.0, "b": 300.0}, uncertainty="delta", storage_root=root,
    )
    assert result.output_property == "temperature"
    assert result.calibration_status == "well_calibrated"
    assert result.tier == "quantitative"  # calibrated → not capped to proxy
    assert result.n_valid > 0

    pm = session.get(PropertyModel, result.model_id)
    assert pm is not None and pm.property == "temperature"
    assert pm.project_id == pid

    reader = open_property_model(pm.store_uri)
    value = reader.read_level("temperature", 0)
    assert value.shape == fused_grid_from_row(fem).shape
    assert reader.has_sigma("temperature")
    sigma = reader.read_sigma_level("temperature", 0)

    # value == a*res + b where finite (read the resampled resistivity off the fused group).
    group_res = _read_fused_resistivity(fem, root)
    finite = np.isfinite(value)
    np.testing.assert_allclose(
        value[finite], (2.0 * group_res + 300.0)[finite], rtol=1e-4, atol=1e-3
    )
    # delta-method σ for y=a*x+b is |a|*σ_x; resampled σ ≈ 0.1*res (near-identity resample).
    sig_x = 0.1 * group_res
    np.testing.assert_allclose(sigma[finite], (2.0 * sig_x)[finite], rtol=0.05, atol=0.5)


def _read_fused_resistivity(fem, root):
    from geosim.fusion.grid import open_fused_group
    group = open_fused_group(fem, storage_root=root)
    layer = [lay for lay in fem.layers if lay.property == "resistivity"][-1]
    return np.asarray(group[layer.id][...], dtype=float)


def test_delta_sigma_numerically_matches_analytic_param_term(env):
    """Param σ on ``a`` adds (∂f/∂a·σ_a)² = (x·σ_a)² to σ_y² (doc 07 §5.2)."""
    session, layout, root, pid = env
    fem, _res = _build_fused_with_resistivity(session, layout, pid, sigma_rel=0.1)

    t = LinearTempFromResistivity()
    t.params = [Param("a", float, default=2.0, range=(0.0, 100.0), sigma=0.5),
                Param("b", float, default=300.0, range=(0.0, 1000.0))]
    result = run_transform(session, layout, fem, t, params={"a": 2.0, "b": 300.0},
                           uncertainty="delta", storage_root=root)
    reader = open_property_model(session.get(PropertyModel, result.model_id).store_uri)
    sigma = reader.read_sigma_level("temperature", 0)

    res = _read_fused_resistivity(fem, root)
    finite = np.isfinite(res)
    sig_x = 0.1 * res
    # σ_y² = (a·σ_x)² + (x·σ_a)²
    expected = np.sqrt((2.0 * sig_x) ** 2 + (res * 0.5) ** 2)
    np.testing.assert_allclose(sigma[finite], expected[finite], rtol=0.05, atol=0.5)


# ─────────────────────────────── NaN mask honesty ───────────────────────────────


def test_nan_mask_honored(env):
    session, layout, root, pid = env
    fem, _res = _build_fused_with_resistivity(session, layout, pid, nan_corner=True)

    result = run_transform(session, layout, fem, LinearTempFromResistivity(),
                           uncertainty="delta", storage_root=root)
    reader = open_property_model(session.get(PropertyModel, result.model_id).store_uri)
    value = reader.read_level("temperature", 0)
    res = _read_fused_resistivity(fem, root)
    # exactly the cells with no resistivity coverage are NaN in the output (no zero-fill).
    assert np.array_equal(np.isnan(value), np.isnan(res))
    assert np.isnan(value).any() and np.isfinite(value).any()


# ─────────────────────────── uncalibrated honesty (proxy) ───────────────────────────


def test_uncalibrated_output_is_proxy_likelihood(env):
    session, layout, root, pid = env
    fem, _res = _build_fused_with_resistivity(session, layout, pid)

    result = run_transform(session, layout, fem, UncalibratedLikelihood(),
                           uncertainty="delta", storage_root=root)
    assert result.calibration_status == "uncalibrated"
    assert result.tier == "proxy"
    assert "likelihood" in result.title.lower()

    # the derivation block records the honesty fields (doc 07 §4.3).
    pm = session.get(PropertyModel, result.model_id)
    prov = session.get(Provenance, session.get(Dataset, pm.dataset_id).provenance_id)
    deriv = json.loads(prov.params_json)["derivation"]
    assert deriv["calibrationStatus"] == "uncalibrated"
    assert deriv["tier"] == "proxy"
    assert "likelihood" in deriv["title"].lower()
    assert deriv["transformId"] == "test.uncal_temp"


# ─────────────────────── Monte-Carlo agrees with delta on linear ───────────────────────


def test_monte_carlo_agrees_with_delta_on_linear(env):
    session, layout, root, pid = env
    fem, _res = _build_fused_with_resistivity(session, layout, pid, sigma_rel=0.1)

    delta = run_transform(session, layout, fem, LinearTempFromResistivity(),
                          params={"a": 2.0, "b": 300.0}, uncertainty="delta",
                          storage_root=root)
    mc = run_transform(session, layout, fem, LinearTempFromResistivity(),
                       params={"a": 2.0, "b": 300.0}, uncertainty="monte_carlo",
                       mc_samples=400, mc_seed=1, storage_root=root)
    assert mc.uncertainty_mode == "monte_carlo"

    d_reader = open_property_model(session.get(PropertyModel, delta.model_id).store_uri)
    m_reader = open_property_model(session.get(PropertyModel, mc.model_id).store_uri)
    d_sig = d_reader.read_sigma_level("temperature", 0)
    m_sig = m_reader.read_sigma_level("temperature", 0)
    d_val = d_reader.read_level("temperature", 0)
    m_val = m_reader.read_level("temperature", 0)
    finite = np.isfinite(d_sig) & np.isfinite(m_sig)
    # linear ⇒ MC mean ≈ delta value, MC std ≈ delta σ (sampling tolerance).
    np.testing.assert_allclose(m_val[finite], d_val[finite], rtol=0.02, atol=2.0)
    np.testing.assert_allclose(m_sig[finite], d_sig[finite], rtol=0.15, atol=1.0)


# ─────────────────────────────── REST endpoints ───────────────────────────────


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
    session.add(Project(id=pid, name="rest", storage_root=str(storage_root)))
    session.add(SpatialFrameRow(
        project_id=pid, mode=frame.mode.value,
        roi_json=json.dumps({"xmin": 0, "xmax": 200, "ymin": 0, "ymax": 200}),
        depth_range_json=json.dumps({"zmin": -200, "zmax": 0}),
        frame_json=json.dumps({"mode": frame.mode.value}),
    ))
    session.commit()
    fem, _res = _build_fused_with_resistivity(session, layout, pid)
    fem_id = fem.id
    session.close()
    yield client, pid, fem_id


def test_endpoint_list_transforms(client_env):
    client, _pid, _fem = client_env
    r = client.get("/transforms")
    assert r.status_code == 200, r.text
    ids = {t["id"] for t in r.json()["transforms"]}
    assert "test.linear_temp" in ids
    spec = next(t for t in r.json()["transforms"] if t["id"] == "test.linear_temp")
    assert spec["target"] == "temperature"
    assert spec["inputs"][0]["name"] == "resistivity"
    assert spec["calibration_status"] == "well_calibrated"


def test_endpoint_transform_sync(client_env):
    client, pid, fem_id = client_env
    r = client.post(f"/fused/{fem_id}/transform", json={
        "project_id": pid, "transform_id": "test.linear_temp",
        "params": {"a": 2.0, "b": 300.0}, "uncertainty": "delta",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "sync"
    assert body["output_property"] == "temperature"
    assert body["tier"] == "quantitative"
    assert body["model_id"] is not None


def test_endpoint_transform_monte_carlo_is_job(client_env):
    client, pid, fem_id = client_env
    r = client.post(f"/fused/{fem_id}/transform", json={
        "project_id": pid, "transform_id": "test.linear_temp",
        "uncertainty": "monte_carlo", "mc_samples": 50,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "job"
    jr = client.get(f"/jobs/{body['job_id']}")
    assert jr.status_code == 200, jr.text
    js = jr.json()
    assert js["status"] == "succeeded", js
    assert js["progress"] == 1.0


def test_endpoint_transform_unknown_is_404(client_env):
    client, pid, fem_id = client_env
    r = client.post(f"/fused/{fem_id}/transform", json={
        "project_id": pid, "transform_id": "nope.not_registered",
    })
    assert r.status_code == 404, r.text
