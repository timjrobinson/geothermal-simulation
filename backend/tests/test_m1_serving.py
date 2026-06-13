"""M1 serving endpoints — meta / zarr / volume / slice (doc 04 §9.2/§9.3, doc 06 §1.3/§12).

Writes a real PropertyModel volume via :mod:`geosim.storage` + a catalog row, then exercises
the endpoints through a FastAPI ``TestClient``:

- ``GET /property-models/{id}`` meta is correct (shape/levels/origin/spacing/stats/colormap),
- ``GET /property-models/{id}/volume`` bytes decode to the right ``(z,y,x)`` shape & values
  (including a NaN no-data cell, doc 02 §10.2) — the M1 single-resident path (doc 06 §1.3),
- ``GET /property-models/{id}/zarr/{path}`` passthrough serves ``zarr.json`` + chunks with
  ETag / Range / immutable caching (doc 04 §9.2),
- ``POST /property-models/{id}/slice`` returns the right plane shape + header (doc 04 §9.3).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from geosim.api import Settings, create_app
from geosim.catalog import IdKind, new_id
from geosim.catalog import Dataset as DatasetRow
from geosim.catalog import Project as ProjectRow
from geosim.catalog import PropertyModel as PropertyModelRow
from geosim.catalog import Provenance as ProvenanceRow
from geosim.catalog import SpatialFrameRow
from geosim.api.frame_io import frame_row_kwargs
from geosim.spatial import SpatialFrame
from geosim.storage import GridSpec, ensure_project_layout, write_property_model

PROPERTY = "resistivity"
ORIGIN = (10.0, 20.0, 30.0)  # (z0, y0, x0) Engineering m
SPACING = (2.0, 3.0, 4.0)  # (dz, dy, dx)


@pytest.fixture
def app_and_ids(tmp_path):
    """An app over a tmp storage root with one written PropertyModel + catalog rows."""
    storage_root = tmp_path / "store"
    settings = Settings(storage_root=storage_root)
    app = create_app(settings)

    pid = new_id(IdKind.PROJECT)
    did = new_id(IdKind.DATASET)
    prov_id = new_id(IdKind.PROVENANCE)
    pm_id = new_id(IdKind.PROPERTY_MODEL)

    layout = ensure_project_layout(storage_root, pid)
    store_path = layout.zarr_path(pm_id)

    # A small (4,5,6) volume with a known ramp + one NaN no-data cell (doc 02 §10.2).
    nz, ny, nx = 4, 5, 6
    values = np.arange(nz * ny * nx, dtype=np.float32).reshape(nz, ny, nx)
    values[1, 2, 3] = np.nan
    sigma = np.ones_like(values)
    write_property_model(
        store_path,
        PROPERTY,
        values,
        grid=GridSpec(origin=ORIGIN, spacing=SPACING),
        sigma=sigma,
        chunk=64,
    )

    Session = app.state.session_factory
    session = Session()
    try:
        project = ProjectRow(id=pid, name="m1", storage_root=str(storage_root))
        project.spatial_frame = SpatialFrameRow(
            project_id=pid, **frame_row_kwargs(SpatialFrame())
        )
        provenance = ProvenanceRow(
            id=prov_id,
            project_id=pid,
            target_kind="propertyModel",
            target_id=pm_id,
            process="ingest:synthetic",
        )
        dataset = DatasetRow(
            id=did,
            project_id=pid,
            name="ds",
            kind="propertyModel",
            method="synthetic",
            status="ready",
            extent_json=json.dumps(
                {"xmin": 0, "ymin": 0, "zmin": 0, "xmax": 1, "ymax": 1, "zmax": 1}
            ),
            spatial_frame_id=pid,
            provenance_id=prov_id,
            version_root_id=did,
            created_by="test",
        )
        pm = PropertyModelRow(
            id=pm_id,
            dataset_id=did,
            project_id=pid,
            property=PROPERTY,
            canonical_unit="ohm.m",
            support="volume",
            store_uri=str(store_path),
            store_format="zarr",
            shape_json=json.dumps([nz, ny, nx]),
            spacing_json=json.dumps(list(SPACING)),
            origin_json=json.dumps(list(ORIGIN)),
            bbox_json=json.dumps(
                {"xmin": 0, "ymin": 0, "zmin": 0, "xmax": 1, "ymax": 1, "zmax": 1}
            ),
            pyramid_levels=1,
        )
        session.add(project)
        session.flush()  # spatial_frame must exist before dataset FK + provenance
        session.add(provenance)
        session.flush()  # provenance must exist before dataset.provenance_id FK
        session.add_all([dataset, pm])
        session.commit()
    finally:
        session.close()

    return app, pid, pm_id, (nz, ny, nx), values


def _decode_volume(resp, shape):
    arr = np.frombuffer(resp.content, dtype="<f4").reshape(shape)
    return arr


def test_meta(app_and_ids):
    app, pid, pm_id, (nz, ny, nx), values = app_and_ids
    client = TestClient(app)
    r = client.get(f"/property-models/{pm_id}")
    assert r.status_code == 200, r.text
    meta = r.json()
    assert meta["id"] == pm_id
    assert meta["property"] == PROPERTY
    assert meta["canonicalUnit"]  # from the property-type registry (doc 01 §5)
    assert meta["shape"] == [nz, ny, nx]
    assert meta["origin"] == list(ORIGIN)
    assert meta["spacing"] == list(SPACING)
    assert meta["levels"] >= 1
    assert meta["colormap"]  # seeded from the property-type registry (doc 01 §5)
    assert meta["scaling"]
    assert meta["hasSigma"] is True
    # NaN-aware stats over the ramp (one NaN cell excluded).
    finite = values[np.isfinite(values)]
    assert meta["stats"]["min"] == pytest.approx(float(finite.min()))
    assert meta["stats"]["max"] == pytest.approx(float(finite.max()))
    assert meta["stats"]["p1"] is not None and meta["stats"]["p99"] is not None
    assert meta["frame"] is not None  # frame summary from the project


def test_meta_404(app_and_ids):
    app, *_ = app_and_ids
    client = TestClient(app)
    assert client.get("/property-models/pm_doesnotexist").status_code == 404


def test_volume_decodes_to_shape_and_values_with_nan(app_and_ids):
    app, pid, pm_id, shape, values = app_and_ids
    client = TestClient(app)
    r = client.get(f"/property-models/{pm_id}/volume?level=0")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/octet-stream"
    assert json.loads(r.headers["x-volume-shape"]) == list(shape)
    assert json.loads(r.headers["x-volume-origin"]) == list(ORIGIN)
    assert json.loads(r.headers["x-volume-spacing"]) == list(SPACING)
    assert r.headers["x-volume-byte-order"] == "little"

    arr = _decode_volume(r, shape)
    # Decoded buffer matches the source volume bit-for-bit, NaN where masked.
    assert np.isnan(arr[1, 2, 3])
    np.testing.assert_array_equal(
        np.nan_to_num(arr, nan=-1.0), np.nan_to_num(values, nan=-1.0)
    )


def test_volume_meta_sidecar(app_and_ids):
    app, pid, pm_id, shape, values = app_and_ids
    client = TestClient(app)
    r = client.get(f"/property-models/{pm_id}/volume/meta?level=0")
    assert r.status_code == 200, r.text
    m = r.json()
    assert m["shape"] == list(shape)
    assert m["origin"] == list(ORIGIN)
    assert m["spacing"] == list(SPACING)
    assert m["noData"] == "NaN"
    assert m["dtype"] == "float32"


def test_volume_bad_level(app_and_ids):
    app, pid, pm_id, *_ = app_and_ids
    client = TestClient(app)
    assert client.get(f"/property-models/{pm_id}/volume?level=99").status_code == 404


def test_zarr_passthrough_group_meta(app_and_ids):
    app, pid, pm_id, *_ = app_and_ids
    client = TestClient(app)
    r = client.get(f"/property-models/{pm_id}/zarr/zarr.json")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/json")
    doc = r.json()
    assert "attributes" in doc or "zarr_format" in doc
    assert r.headers.get("etag")


def test_zarr_passthrough_chunk_immutable_and_etag(app_and_ids):
    app, pid, pm_id, *_ = app_and_ids
    client = TestClient(app)
    chunk_path = f"{PROPERTY}/0/c/0/0/0"
    r = client.get(f"/property-models/{pm_id}/zarr/{chunk_path}")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/octet-stream"
    assert "immutable" in r.headers.get("cache-control", "")
    etag = r.headers["etag"]
    assert etag
    # If-None-Match → 304 (doc 04 §9.2).
    r304 = client.get(
        f"/property-models/{pm_id}/zarr/{chunk_path}",
        headers={"If-None-Match": etag},
    )
    assert r304.status_code == 304


def test_zarr_passthrough_range(app_and_ids):
    app, pid, pm_id, *_ = app_and_ids
    client = TestClient(app)
    chunk_path = f"{PROPERTY}/0/c/0/0/0"
    full = client.get(f"/property-models/{pm_id}/zarr/{chunk_path}").content
    r = client.get(
        f"/property-models/{pm_id}/zarr/{chunk_path}",
        headers={"Range": "bytes=0-3"},
    )
    assert r.status_code == 206, r.text
    assert r.content == full[:4]
    assert r.headers["content-range"].startswith("bytes 0-3/")


def test_zarr_passthrough_traversal_blocked(app_and_ids):
    app, pid, pm_id, *_ = app_and_ids
    client = TestClient(app)
    r = client.get(f"/property-models/{pm_id}/zarr/../../../etc/passwd")
    assert r.status_code in (403, 404)


def test_slice_z_plane(app_and_ids):
    app, pid, pm_id, shape, values = app_and_ids
    nz, ny, nx = shape
    client = TestClient(app)
    r = client.post(
        f"/property-models/{pm_id}/slice",
        json={"plane": "z", "position": 1, "level": 0, "encoding": "f32"},
    )
    assert r.status_code == 200, r.text
    header = json.loads(r.headers["x-slice-header"])
    assert header["width"] == nx and header["height"] == ny
    assert header["dx"] == SPACING[2] and header["dy"] == SPACING[1]
    img = np.frombuffer(r.content, dtype="<f4").reshape(header["height"], header["width"])
    expected = values[1, :, :]
    np.testing.assert_array_equal(
        np.nan_to_num(img, nan=-1.0), np.nan_to_num(expected, nan=-1.0)
    )


def test_slice_x_plane_shape(app_and_ids):
    app, pid, pm_id, shape, values = app_and_ids
    nz, ny, nx = shape
    client = TestClient(app)
    r = client.post(
        f"/property-models/{pm_id}/slice",
        json={"plane": "x", "position": 2, "level": 0},
    )
    assert r.status_code == 200, r.text
    header = json.loads(r.headers["x-slice-header"])
    # constant-x → (z, y) image
    assert header["width"] == ny and header["height"] == nz
    img = np.frombuffer(r.content, dtype="<f4").reshape(header["height"], header["width"])
    np.testing.assert_array_equal(img, values[:, :, 2])


def test_slice_out_of_range_and_bad_encoding(app_and_ids):
    app, pid, pm_id, shape, _ = app_and_ids
    nz, ny, nx = shape
    client = TestClient(app)
    assert (
        client.post(
            f"/property-models/{pm_id}/slice",
            json={"plane": "z", "position": 999},
        ).status_code
        == 400
    )
    assert (
        client.post(
            f"/property-models/{pm_id}/slice",
            json={"plane": "z", "position": 0, "encoding": "png"},
        ).status_code
        == 400
    )
