"""Tests for the fusion core (doc 07 §1–§2, doc 02 §11).

Small/coarse grids, local temp dirs, SQLite in-memory — no Docker/Postgres/Redis.

Builds two native PropertyModels on different spacings/footprints (resistivity, log10
interp space; density, linear), creates a fused grid (auto-resolution sane), resamples
both in, and checks:

- co-located arrays on the SAME fused grid,
- NaN outside each native footprint (footprint honesty, doc 07 §2.3),
- σ present and inflated where the fused grid upsamples (doc 07 §5.2),
- the native originals are never modified (doc 07 §2.1),
- ``GET /projects/{pid}/artifacts`` lists the models.
"""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from geosim.api.app import Settings, create_app
from geosim.catalog import (
    Dataset,
    FusedLayer,
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
    DEFAULT_CELL_CAP,
    auto_resolution,
    build_fused_model,
    fused_grid_from_row,
    resample_to_fused,
)
from geosim.spatial import Aabb, DepthRange, SpatialFrame
from geosim.storage import (
    GridSpec,
    ensure_project_layout,
    open_property_model,
    write_property_model,
)

# ───────────────────────────────── fixtures ─────────────────────────────────


@pytest.fixture
def env(tmp_path):
    """In-memory catalog + temp storage + a seeded project frame."""
    engine = make_engine()
    create_all(engine)
    Session = session_factory(engine)
    session = Session()
    storage_root = tmp_path
    pid = new_id(IdKind.PROJECT)
    layout = ensure_project_layout(storage_root, pid)
    frame = SpatialFrame(roi=Aabb(0, 1000, 0, 1000), depth_range=DepthRange(-500, 0))
    session.add(Project(id=pid, name="fuse-test", storage_root=str(storage_root)))
    session.add(SpatialFrameRow(
        project_id=pid, mode=frame.mode.value,
        roi_json=json.dumps({"xmin": 0, "xmax": 1000, "ymin": 0, "ymax": 1000}),
        depth_range_json=json.dumps({"zmin": -500, "zmax": 0}),
        frame_json=json.dumps({"mode": frame.mode.value}),
    ))
    session.commit()
    yield session, layout, storage_root, pid
    session.close()


def _aabb(xmin, xmax, ymin, ymax, zmin, zmax) -> str:
    return json.dumps({
        "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "zmin": zmin, "zmax": zmax
    })


def _make_native_pm(
    session, layout, storage_root, pid, *, prop, values, origin, spacing, sigma=None, method="mt"
):
    """Write a native PropertyModel Zarr + catalog rows; return the row."""
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
        id=pm_id, dataset_id=ds_id, project_id=pid, property=prop,
        canonical_unit="ohm*m" if prop == "resistivity" else "kg/m**3", support="volume",
        store_uri=str(zarr_path), shape_json=json.dumps([nz, ny, nx]),
        spacing_json=json.dumps(list(spacing)), origin_json=json.dumps(list(origin)),
        bbox_json=bbox,
        uncertainty_uri=(f"{prop}_sigma" if sigma is not None else None),
    )
    session.add(row)
    session.commit()
    return row


# ───────────────────────────────── auto-resolution ─────────────────────────────────


def test_auto_resolution_clamped_and_capped():
    # median native spacing 50 m over a 1000 m cube → within [1000/512, 1000/64] = [~2, ~16]?
    # 50 > 15.6 (hi clamp) so it clamps to ~15.6 m.
    sp = auto_resolution([40.0, 50.0, 60.0], (1000.0, 1000.0, 1000.0))
    assert sp[0] == sp[1] == sp[2]  # isotropic (anisotropy off by default)
    hi = 1000.0 / 64.0
    lo = 1000.0 / 512.0
    assert lo <= sp[0] <= hi + 1e-9

    # cell cap honoured: a tiny spacing on a big extent is coarsened under the cap.
    sp2 = auto_resolution([0.1], (10000.0, 10000.0, 10000.0), cell_cap=DEFAULT_CELL_CAP)
    nz = int(np.ceil(10000.0 / sp2[0]) + 1)
    assert nz**3 <= DEFAULT_CELL_CAP


# ───────────────────────────────── grid build ─────────────────────────────────


def test_build_fused_model_auto_resolution_sane(env):
    session, layout, storage_root, pid = env
    res = _make_native_pm(
        session, layout, storage_root, pid, prop="resistivity",
        values=np.full((4, 4, 4), 100.0, dtype=np.float32),
        origin=(-500.0, 0.0, 0.0), spacing=(100.0, 250.0, 250.0),
    )
    fem, grid = build_fused_model(session, layout, pid, source_property_model_ids=[res.id])
    assert fem.id.startswith("fem_")
    assert grid.n_cells <= DEFAULT_CELL_CAP
    assert grid.n_cells > 0
    # The grid is a container — its bbox spans the source footprint.
    box = json.loads(fem.bbox_json)
    assert box["xmin"] == pytest.approx(0.0)
    assert box["zmin"] == pytest.approx(-500.0)
    # catalog rows written: dataset(kind=fusedModel) + fused_models row reachable.
    assert session.get(Dataset, fem.dataset_id).kind == "fusedModel"


# ───────────────────────────────── resample both ─────────────────────────────────


def test_resample_two_models_colocated_footprint_and_sigma(env):
    session, layout, storage_root, pid = env

    # Resistivity: orders-of-magnitude (log10 interp), footprint = left half (x in [0,400]).
    res_vals = np.empty((4, 4, 5), dtype=np.float32)
    res_vals[:] = np.linspace(10.0, 1000.0, 5)[None, None, :]  # spans 2 decades along x
    res = _make_native_pm(
        session, layout, storage_root, pid, prop="resistivity",
        values=res_vals, origin=(-400.0, 0.0, 0.0), spacing=(100.0, 100.0, 100.0),
    )

    # Density: linear, DIFFERENT spacing + footprint (x in [200,800], shifted), with sigma.
    den_vals = np.full((3, 3, 4), 2500.0, dtype=np.float32)
    den_sigma = np.full((3, 3, 4), 50.0, dtype=np.float32)
    den = _make_native_pm(
        session, layout, storage_root, pid, prop="density", method="gravity",
        values=den_vals, sigma=den_sigma,
        origin=(-300.0, 200.0, 200.0), spacing=(150.0, 200.0, 200.0),
    )

    fem, grid = build_fused_model(
        session, layout, pid,
        source_property_model_ids=[res.id, den.id], spacing=(50.0, 50.0, 50.0),
    )

    res_ref = resample_to_fused(session, fem, res.id, storage_root=storage_root)
    den_ref = resample_to_fused(session, fem, den.id, storage_root=storage_root)

    assert res_ref.interp_space == "log10"  # from the registry
    assert den_ref.interp_space == "linear"

    group = fused_grid_from_row(fem)
    from geosim.fusion import open_fused_group
    z = open_fused_group(fem, storage_root=storage_root)
    res_arr = np.asarray(z[res_ref.value_array])
    den_arr = np.asarray(z[den_ref.value_array])
    res_sig = np.asarray(z[res_ref.sigma_array])
    den_sig = np.asarray(z[den_ref.sigma_array])
    res_mask = np.asarray(z[res_ref.coverage_mask])
    den_mask = np.asarray(z[den_ref.coverage_mask])

    # Co-located: same shape, same grid.
    assert res_arr.shape == den_arr.shape == group.shape

    # Footprint honesty: each layer is NaN outside its own native footprint, and the two
    # footprints differ (different x ranges), so each has cells the other lacks.
    res_cov = np.isfinite(res_arr)
    den_cov = np.isfinite(den_arr)
    assert res_cov.any() and den_cov.any()
    assert (res_cov & ~den_cov).any(), "resistivity must cover cells density does not"
    assert (den_cov & ~res_cov).any(), "density must cover cells resistivity does not"
    # masks match the finite pattern.
    assert np.array_equal(res_mask > 0.5, res_cov)
    assert np.array_equal(den_mask > 0.5, den_cov)

    # No extrapolation: density is NaN well outside its footprint (e.g. x near 0).
    fz, fy, fx = group.axis_coords()
    x_near0 = int(np.argmin(np.abs(fx - 0.0)))
    assert np.all(~np.isfinite(den_arr[:, :, x_near0]))

    # Sigma present everywhere the value is, and finite.
    assert np.all(np.isfinite(res_sig[res_cov]))
    assert np.all(np.isfinite(den_sig[den_cov]))

    # Sigma inflated where upsampled: the fused grid (50 m) is finer than resistivity's
    # native 100 m → interpolation-variance inflates σ above the bare resampled native σ.
    # Compare against the default rel-σ * value (resistivity had no native σ).
    bare = np.abs(res_arr) * 0.15  # default_rel_sigma for resistivity (registry)
    inflated_somewhere = np.any(res_sig[res_cov] > bare[res_cov] + 1e-6)
    assert inflated_somewhere, "interpolation-variance must inflate σ where upsampled"

    # Resampled values sane: within native min/max range (log-space interpolation keeps
    # them positive and bounded by the native decade span).
    finite_res = res_arr[res_cov]
    assert finite_res.min() >= 10.0 - 1e-3
    assert finite_res.max() <= 1000.0 + 1.0


def test_originals_untouched_by_resample(env):
    session, layout, storage_root, pid = env
    vals = np.linspace(50.0, 500.0, 2 * 3 * 4).reshape(2, 3, 4).astype(np.float32)
    res = _make_native_pm(
        session, layout, storage_root, pid, prop="resistivity",
        values=vals, origin=(-200.0, 0.0, 0.0), spacing=(100.0, 100.0, 100.0),
    )
    before = open_property_model(res.store_uri).read_level("resistivity", 0).copy()
    fem, _ = build_fused_model(session, layout, pid, source_property_model_ids=[res.id])
    resample_to_fused(session, fem, res.id, storage_root=storage_root)
    after = open_property_model(res.store_uri).read_level("resistivity", 0)
    assert np.array_equal(before, after), "native original must never be modified (doc 07 §2.1)"


def test_resample_is_cached(env):
    session, layout, storage_root, pid = env
    res = _make_native_pm(
        session, layout, storage_root, pid, prop="resistivity",
        values=np.full((3, 3, 3), 200.0, dtype=np.float32),
        origin=(-200.0, 0.0, 0.0), spacing=(100.0, 100.0, 100.0),
    )
    fem, _ = build_fused_model(session, layout, pid, source_property_model_ids=[res.id])
    a = resample_to_fused(session, fem, res.id, storage_root=storage_root)
    b = resample_to_fused(session, fem, res.id, storage_root=storage_root)
    assert a.cached is False
    assert b.cached is True
    assert b.layer_id == a.layer_id
    # exactly one FusedLayer row for this (pm, grid, method) key.
    n = session.query(FusedLayer).filter(FusedLayer.fused_model_id == fem.id).count()
    assert n == 1


# ───────────────────────────────── API surface ─────────────────────────────────


def test_fusion_api_create_resample_get_and_artifacts(tmp_path):
    settings = Settings(storage_root=tmp_path)
    app = create_app(settings)
    client = TestClient(app)

    # Build a project via the API.
    r = client.post("/projects", json={"name": "api-fuse", "frame": {
        "mode": "local",
        "roi": {"xmin": 0, "xmax": 1000, "ymin": 0, "ymax": 1000},
        "depth_range": {"zmin": -500, "zmax": 0},
    }})
    assert r.status_code == 201, r.text
    pid = r.json()["id"]

    # Seed two native PMs directly in the catalog/storage (ingestion is another component).
    Session = app.state.session_factory
    session = Session()
    layout = ensure_project_layout(tmp_path, pid)
    res = _make_native_pm(
        session, layout, tmp_path, pid, prop="resistivity",
        values=np.full((3, 3, 4), 100.0, dtype=np.float32),
        origin=(-300.0, 0.0, 0.0), spacing=(100.0, 100.0, 100.0),
    )
    den = _make_native_pm(
        session, layout, tmp_path, pid, prop="density", method="gravity",
        values=np.full((3, 3, 3), 2600.0, dtype=np.float32),
        origin=(-300.0, 100.0, 100.0), spacing=(150.0, 150.0, 150.0),
    )
    session.close()

    # POST /fused
    r = client.post(
        "/fused", json={"project_id": pid, "source_property_model_ids": [res.id, den.id]}
    )
    assert r.status_code == 201, r.text
    fem = r.json()
    grid_id = fem["id"]
    assert fem["grid_type"] == "regular_voxel"
    assert fem["n_cells"] <= DEFAULT_CELL_CAP and fem["n_cells"] > 0

    # POST /fused/{gridId}/resample for both
    r1 = client.post(f"/fused/{grid_id}/resample", json={"property_model_id": res.id})
    r2 = client.post(f"/fused/{grid_id}/resample", json={"property_model_id": den.id})
    assert r1.status_code == 201 and r2.status_code == 201, (r1.text, r2.text)
    assert r1.json()["interp_space"] == "log10"

    # GET /fused/{gridId} lists both layers.
    r = client.get(f"/fused/{grid_id}")
    assert r.status_code == 200
    props = {lay["property"] for lay in r.json()["layers"]}
    assert props == {"resistivity", "density"}

    # GET /projects/{pid}/artifacts lists the native PMs + the fused model.
    r = client.get(f"/projects/{pid}/artifacts")
    assert r.status_code == 200, r.text
    arts = r.json()
    ids = {a["id"] for a in arts}
    assert res.id in ids and den.id in ids
    assert any(a["kind"] == "fusedModel" for a in arts)

    # property filter
    r = client.get(f"/projects/{pid}/artifacts", params={"property": "resistivity"})
    rids = {a["id"] for a in r.json()}
    assert res.id in rids and den.id not in rids

    # method filter
    r = client.get(f"/projects/{pid}/artifacts", params={"method": "gravity"})
    mids = {a["id"] for a in r.json()}
    assert den.id in mids and res.id not in mids

    # bbox filter restricts to a far-away empty box.
    r = client.get(f"/projects/{pid}/artifacts", params={"bbox": "9000,9100,9000,9100,9000,9100"})
    assert r.json() == []
