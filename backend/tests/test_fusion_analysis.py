"""Tests for fusion cross-plot / stats / clustering (doc 07 §3).

Small/coarse grids, local temp dirs, SQLite in-memory — no Docker/Postgres/Redis.

Builds two co-located native PropertyModels (resistivity, log10 interp space; density,
linear) over the SAME footprint with a **planted anomaly** (a low-resistivity / low-density
blob), resamples both into a small fused grid, then checks the §3 analysis surface:

- co-located sampling → a listwise feature matrix on cells where ALL layers are present,
  plus a region-of-interest bbox clip (doc 07 §3.1);
- a cross-plot payload (scatter point set + correlation matrix + histogram) shape (§3.2);
- k-means AND GMM separate the planted anomaly into its own cluster — the M4 exit
  assertion in miniature (§3.3);
- clustering writes back a categorical ``lithology_class`` class PropertyModel (+ GMM
  per-class probability volumes), stored & catalogued like any derived volume (§4.3);
- a cross-plot selection maps back to a boolean volume mask (linked brushing, §3.2);
- the REST endpoints (sample / crossplot / cluster) return the expected shapes.
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
    build_fused_model,
    cluster_fused,
    correlation_matrix,
    crossplot,
    fused_grid_from_row,
    histogram,
    resample_to_fused,
    sample_fused,
    selection_to_mask,
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
    frame = SpatialFrame(roi=Aabb(0, 200, 0, 200), depth_range=DepthRange(-200, 0))
    session.add(Project(id=pid, name="fuse-analysis-test", storage_root=str(storage_root)))
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


def _make_native_pm(session, layout, pid, *, prop, values, origin, spacing, unit, method="mt"):
    ds_id = new_id(IdKind.DATASET)
    pm_id = new_id(IdKind.PROPERTY_MODEL)
    prov_id = new_id(IdKind.PROVENANCE)
    zarr_path = layout.zarr_path(pm_id)
    grid = GridSpec(origin=origin, spacing=spacing, cell_ref="center")
    write_property_model(zarr_path, prop, values, grid=grid, overwrite=True)

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


def _planted_fields(shape=(8, 8, 8)):
    """Two co-located fields with a planted low-resistivity / low-density anomaly blob.

    Background: resistivity ~100 ohm·m, density ~2700 kg/m³.
    Anomaly (a corner octant): resistivity ~5 ohm·m, density ~2200 kg/m³ — a clearly
    separable second population for clustering.
    """
    nz, ny, nx = shape
    rng = np.random.default_rng(42)
    res = np.full(shape, 100.0) + rng.normal(0, 2.0, shape)
    den = np.full(shape, 2700.0) + rng.normal(0, 10.0, shape)
    # planted blob in the lower-near corner octant
    res[: nz // 2, : ny // 2, : nx // 2] = 5.0 + rng.normal(0, 0.3, (nz // 2, ny // 2, nx // 2))
    den[: nz // 2, : ny // 2, : nx // 2] = 2200.0 + rng.normal(0, 5.0, (nz // 2, ny // 2, nx // 2))
    return res, den


def _build_fused_with_two_layers(session, layout, pid):
    """Resistivity (log10) + density (linear) resampled into a small fused grid."""
    res, den = _planted_fields((8, 8, 8))
    origin = (-160.0, 20.0, 20.0)
    spacing = (20.0, 20.0, 20.0)
    res_pm = _make_native_pm(
        session, layout, pid, prop="resistivity", values=res, origin=origin,
        spacing=spacing, unit="ohm*m", method="mt",
    )
    den_pm = _make_native_pm(
        session, layout, pid, prop="density", values=den, origin=origin,
        spacing=spacing, unit="kg/m**3", method="gravity",
    )
    fem, _grid = build_fused_model(
        session, layout, pid,
        source_property_model_ids=[res_pm.id, den_pm.id],
        spacing=(20.0, 20.0, 20.0), name="fused-analysis",
    )
    resample_to_fused(session, fem, res_pm.id)
    resample_to_fused(session, fem, den_pm.id)
    session.refresh(fem)
    return fem, res_pm, den_pm


# ───────────────────────────────── co-located sampling (§3.1) ─────────────────────────────────


def test_sample_listwise_feature_matrix(env):
    session, layout, _root, pid = env
    fem, _r, _d = _build_fused_with_two_layers(session, layout, pid)

    s = sample_fused(session, fem, ["resistivity", "density"], mode="all")
    assert s.properties == ["resistivity", "density"]
    assert s.features.shape[1] == 2
    assert s.n > 0
    # listwise: every retained row is fully finite (doc 07 §3.1).
    assert np.isfinite(s.features).all()
    # cell_index/coords align with the feature rows.
    assert s.cell_index.shape[0] == s.n
    assert s.coords.shape == (s.n, 3)


def test_sample_bbox_clip_restricts_working_set(env):
    session, layout, _root, pid = env
    fem, _r, _d = _build_fused_with_two_layers(session, layout, pid)

    full = sample_fused(session, fem, ["resistivity", "density"], mode="all")
    grid = fused_grid_from_row(fem)
    bb = grid.bbox()
    # clip to the lower half in z → fewer cells than the full set.
    zmid = 0.5 * (bb["zmin"] + bb["zmax"])
    clip = dict(bb, zmax=zmid)
    sub = sample_fused(session, fem, ["resistivity", "density"], mode="all", bbox=clip)
    assert 0 < sub.n < full.n


# ───────────────────────────────── cross-plot / stats (§3.2) ─────────────────────────────────


def test_crossplot_scatter_and_correlation(env):
    session, layout, _root, pid = env
    fem, _r, _d = _build_fused_with_two_layers(session, layout, pid)
    s = sample_fused(session, fem, ["resistivity", "density"], mode="all")

    cp = crossplot(s, ["resistivity", "density"], color_by="depth")
    assert cp["kind"] == "scatter"
    assert cp["axes"] == ["resistivity", "density"]
    assert len(cp["points"]) == s.n
    assert len(cp["color"]) == s.n  # per-point depth colour

    corr = correlation_matrix(s)
    assert corr["properties"] == ["resistivity", "density"]
    m = corr["matrix"]
    assert len(m) == 2 and len(m[0]) == 2
    # resistivity & density both drop in the anomaly → positively correlated.
    assert m[0][1] is not None and m[0][1] > 0.3


def test_histogram_shape(env):
    session, layout, _root, pid = env
    fem, _r, _d = _build_fused_with_two_layers(session, layout, pid)
    s = sample_fused(session, fem, ["resistivity", "density"], mode="all")
    h = histogram(s, "resistivity", bins=16, kde=True)
    assert len(h["counts"]) == 16
    assert len(h["bin_edges"]) == 17
    assert h["n"] == s.n
    assert "kde_x" in h and len(h["kde_x"]) == 16


# ─────────────────── clustering (§3.3) — M4 exit assertion in miniature ───────────────────


def _anomaly_separated(labels: np.ndarray, sample) -> bool:
    """The planted low-resistivity cells fall predominantly in ONE cluster."""
    res = sample.features[:, sample.properties.index("resistivity")]
    anomaly = res < 20.0  # the blob is ~5 ohm·m, background ~100
    assert anomaly.any(), "fixture must contain anomaly cells"
    anomaly_labels = labels[anomaly]
    # the dominant label among anomaly cells should cover almost all of them...
    dominant = np.bincount(anomaly_labels).argmax()
    purity = np.mean(anomaly_labels == dominant)
    # ...and background cells should rarely carry that label (well-separated cluster).
    bg_share = np.mean(labels[~anomaly] == dominant)
    return purity > 0.9 and bg_share < 0.1


@pytest.mark.parametrize("algorithm", ["kmeans", "gmm"])
def test_clustering_separates_anomaly_and_writes_class_volume(env, algorithm):
    session, layout, _root, pid = env
    fem, _r, _d = _build_fused_with_two_layers(session, layout, pid)

    result = cluster_fused(
        session, layout, fem, properties=["resistivity", "density"],
        algorithm=algorithm, n_clusters=2, write_volumes=True,
    )
    assert result.algorithm == algorithm
    assert result.n_clusters == 2
    assert len(result.centroids) == 2
    assert sum(result.sizes) == result.labels.shape[0]

    s = sample_fused(session, fem, ["resistivity", "density"], mode="all")
    assert _anomaly_separated(result.labels, s), "clustering must isolate the planted anomaly"

    # a categorical class PropertyModel was written + catalogued (doc 07 §3.3 step 5 / §4.3).
    assert result.class_model_id is not None
    class_pm = session.get(PropertyModel, result.class_model_id)
    assert class_pm is not None
    assert class_pm.property == "lithology_class"
    assert class_pm.project_id == pid
    reader = open_property_model(class_pm.store_uri)
    vol = reader.read_level("lithology_class", 0)
    assert vol.shape == fused_grid_from_row(fem).shape
    # at least two distinct class labels present in the volume.
    finite = vol[np.isfinite(vol)]
    assert np.unique(np.round(finite)).size >= 2

    if algorithm == "gmm":
        # one per-class probability volume per cluster (doc 07 §3.3 step 5).
        assert len(result.probability_model_ids) == 2
        prob_pm = session.get(PropertyModel, result.probability_model_ids[0])
        assert prob_pm is not None and prob_pm.property == "lithology_class"


def test_selection_to_mask_round_trip(env):
    session, layout, _root, pid = env
    fem, _r, _d = _build_fused_with_two_layers(session, layout, pid)
    s = sample_fused(session, fem, ["resistivity", "density"], mode="all")

    # select the low-resistivity anomaly points in the cross-plot.
    res = s.features[:, s.properties.index("resistivity")]
    sel = np.flatnonzero(res < 20.0)
    mask = selection_to_mask(s, sel)
    assert mask.shape == fused_grid_from_row(fem).shape
    assert mask.dtype == bool
    assert int(mask.sum()) == sel.size


# ───────────────────────────────── REST endpoints ─────────────────────────────────


@pytest.fixture
def client_env(tmp_path):
    """A TestClient over a real app + a project with a fused grid + two layers."""
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
    fem, _r, _d = _build_fused_with_two_layers(session, layout, pid)
    fem_id = fem.id
    session.close()
    yield client, pid, fem_id


def test_endpoint_sample(client_env):
    client, _pid, fem_id = client_env
    r = client.post(f"/fused/{fem_id}/sample",
                    json={"properties": ["resistivity", "density"], "mode": "all"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["properties"] == ["resistivity", "density"]
    assert body["n"] == len(body["features"]) == len(body["cell_index"])


def test_endpoint_crossplot(client_env):
    client, _pid, fem_id = client_env
    r = client.post(f"/fused/{fem_id}/crossplot", json={
        "axes": ["resistivity", "density"], "color_by": "depth",
        "histogram_property": "resistivity", "correlation": True,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["crossplot"]["kind"] == "scatter"
    assert "histogram" in body
    assert "correlation" in body


def test_endpoint_cluster_sync(client_env):
    client, pid, fem_id = client_env
    r = client.post(f"/fused/{fem_id}/cluster", json={
        "project_id": pid, "algorithm": "kmeans", "n_clusters": 2, "write_volumes": True,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "sync"
    assert body["n_clusters"] == 2
    assert body["class_model_id"] is not None


def test_endpoint_cluster_job(client_env):
    client, pid, fem_id = client_env
    # force the job path (the small grid is well under the sync limit).
    r = client.post(f"/fused/{fem_id}/cluster", json={
        "project_id": pid, "algorithm": "gmm", "n_clusters": 2, "force_job": True,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "job"
    job_id = body["job_id"]
    # InlineJobRunner runs synchronously → the job is already terminal.
    jr = client.get(f"/jobs/{job_id}")
    assert jr.status_code == 200, jr.text
    js = jr.json()
    assert js["status"] == "succeeded", js
    assert js["progress"] == 1.0
