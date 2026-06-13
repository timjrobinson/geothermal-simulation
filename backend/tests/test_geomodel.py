"""M8 implicit geomodel tests (OVERVIEW §6 L4 / doc 07; doc 02 §5/§10.2).

A SMALL/coarse GemPy model (2 stratigraphic layers + 1 fault) on the numpy backend,
asserting the three deliverables:

1. :func:`build_geomodel` produces a per-cell lithology over the regular grid (doc 07),
   in doc-02 ``[z, y, x]`` Z-up order, with a normalised class-probability axis (doc 02
   §10.2);
2. :func:`persist_geomodel` writes + catalogs a categorical ``lithology_class``
   PropertyModel with a categories attr table (doc 02 §10.2); and
3. at least one ``unitSolid`` GeologicalFeature with a valid ``.glb`` is created (doc 02 §5),
   plus the ``POST /projects/{pid}/geomodel`` API surface returns those ids.

Kept fast: a ``≤15³`` grid + a handful of interface/orientation points (CLAUDE.md).
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest
from fastapi.testclient import TestClient

from geosim.api import Settings, create_app
from geosim.api.frame_io import frame_row_kwargs
from geosim.catalog import (
    Feature,
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
from geosim.geomodel import (
    GeoModelResult,
    build_geomodel,
    persist_geomodel,
    spec_from_catalog_surfaces,
)
from geosim.spatial import Aabb, DepthRange, SpatialFrame
from geosim.storage import ensure_project_layout, open_property_model

_GLB_MAGIC = 0x46546C67  # "glTF"


def _frame() -> SpatialFrame:
    return SpatialFrame(roi=Aabb(0, 1000, 0, 1000), depth_range=DepthRange(0, 500))


def _surfaces() -> list[dict]:
    """A 2-layer + 1-fault setup (a handful of interface points each)."""
    return [
        {"name": "layer1", "kind": "horizon",
         "points": [[100, 100, 300], [500, 500, 300], [900, 900, 300],
                    [100, 900, 300], [900, 100, 300]]},
        {"name": "layer2", "kind": "horizon",
         "points": [[100, 100, 150], [500, 500, 150], [900, 900, 150],
                    [100, 900, 150], [900, 100, 150]]},
        {"name": "fault1", "kind": "fault",
         "points": [[500, 100, 100], [500, 500, 250], [500, 900, 400],
                    [520, 300, 150], [520, 700, 350]]},
    ]


@pytest.fixture(scope="module")
def built() -> GeoModelResult:
    """Build the small model once (module-scoped — the GemPy compute is the slow bit)."""
    spec = spec_from_catalog_surfaces(_frame(), _surfaces(), resolution=(12, 12, 12))
    return build_geomodel(spec)


# ─────────────────── deliverable 1: per-cell lithology + class prob ───────────────────
def test_build_produces_per_cell_lithology(built: GeoModelResult) -> None:
    """A per-cell lithology block over the regular grid, [z,y,x] Z-up (doc 07 / doc 02 §10.2)."""
    assert built.lith_zyx.shape == (12, 12, 12)
    labels = np.unique(np.round(built.lith_zyx)).astype(int)
    # ≥2 distinct stratigraphic units are present over the grid (a layered model).
    assert labels.size >= 2
    # Every label decodes to a category in the table.
    cat_ids = {c["id"] for c in built.categories}
    assert set(labels.tolist()).issubset(cat_ids)


def test_class_probability_axis_sums_to_one(built: GeoModelResult) -> None:
    """The class-probability axis is normalised across classes (doc 02 §10.2)."""
    n_class = built.class_prob.shape[0]
    assert n_class == len([c for c in built.categories if not c["isFault"]])
    sums = built.class_prob.sum(axis=0)
    assert np.allclose(sums, 1.0, atol=1e-4)
    assert built.class_prob.min() >= 0.0


def test_fault_in_categories(built: GeoModelResult) -> None:
    """The fault contact is catalogued + flagged isFault (doc 02 §5)."""
    faults = [c for c in built.categories if c["isFault"]]
    assert any(c["name"] == "fault1" for c in faults)


# ───────────────────────────── deliverable 2 + 3: catalog the result ─────────────────────────────
@pytest.fixture
def env(tmp_path):
    engine = make_engine()
    create_all(engine)
    session = session_factory(engine)()
    pid = new_id(IdKind.PROJECT)
    layout = ensure_project_layout(tmp_path, pid)
    session.add(Project(id=pid, name="geomodel-test", storage_root=str(tmp_path)))
    session.add(SpatialFrameRow(project_id=pid, **frame_row_kwargs(_frame())))
    session.commit()
    yield session, layout, pid
    session.close()


def test_persist_writes_lithology_model_and_unit_solids(env, built: GeoModelResult) -> None:
    """A lithology_class PropertyModel + categories table + ≥1 valid unitSolid glb (doc 02)."""
    session, layout, pid = env
    persisted = persist_geomodel(
        session, layout, pid, built,
        extent=_frame_extent(), created_by="test",
    )

    # --- the categorical lithology PropertyModel (doc 02 §10.2) ---
    pm = session.get(PropertyModel, persisted.property_model_id)
    assert pm is not None
    assert pm.property == "lithology_class"
    assert pm.support == "volume"
    assert json.loads(pm.shape_json) == list(built.shape_zyx)

    reader = open_property_model(persisted.lithology_store_uri)
    assert "lithology_class" in reader.properties
    attrs = reader.attrs("lithology_class")
    categories = attrs["categories"]
    assert categories and all({"index", "id", "name"} <= set(c) for c in categories)
    # the stored hard labels round-trip as integer-valued floats
    level0 = reader.read_level("lithology_class")
    assert level0.shape == built.shape_zyx
    assert np.allclose(level0, np.round(level0))

    # the class-probability axis is recorded alongside (doc 02 §10.2)
    cp = reader.group["lithology_class"].attrs["classProbability"]
    assert cp["axis"] == "class" and cp["nClasses"] == built.class_prob.shape[0]

    # --- ≥1 unitSolid feature with a valid glTF (doc 02 §5) ---
    assert persisted.unit_solid_feature_ids
    feats = session.query(Feature).filter_by(project_id=pid).all()
    assert feats and all(f.feature_type == "unitSolid" for f in feats)
    glb = open(feats[0].store_uri, "rb").read()
    magic, version, total = struct.unpack("<III", glb[:12])
    assert magic == _GLB_MAGIC
    assert version == 2
    assert total == len(glb)  # header total matches the file length (valid glb)
    props = json.loads(feats[0].props_json)
    assert props["triangleCount"] > 0 and props["vertexCount"] > 0

    # --- provenance lineage exists (doc 02 §7) ---
    prov = session.get(Provenance, persisted.provenance_id)
    assert prov is not None and prov.process == "model:gempy-implicit"


# ───────────────────────────── deliverable: API surface ─────────────────────────────
def test_post_geomodel_endpoint(tmp_path) -> None:
    """POST /projects/{pid}/geomodel builds + returns the feature + lithology-model ids."""
    app = create_app(Settings(storage_root=tmp_path))
    client = TestClient(app)
    pid = client.post(
        "/projects",
        json={"name": "api-geomodel", "frame": {
            "mode": "local",
            "roi": {"xmin": 0, "xmax": 1000, "ymin": 0, "ymax": 1000},
            "depth_range": {"zmin": 0, "zmax": 500},
        }},
    ).json()["id"]

    resp = client.post(
        f"/projects/{pid}/geomodel",
        json={"surfaces": _surfaces(), "resolution": [10, 10, 10]},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["propertyModelId"].startswith("pm_")
    assert body["unitSolidFeatureIds"]
    assert body["shape"] == [10, 10, 10]
    assert any(c["isFault"] for c in body["categories"])

    # the catalogued PropertyModel is fetchable via the M1 read surface
    meta = client.get(f"/property-models/{body['propertyModelId']}")
    assert meta.status_code == 200
    assert meta.json()["property"] == "lithology_class"


def test_post_geomodel_404_unknown_project(tmp_path) -> None:
    app = create_app(Settings(storage_root=tmp_path))
    client = TestClient(app)
    resp = client.post(
        "/projects/prj_does_not_exist/geomodel",
        json={"surfaces": _surfaces()},
    )
    assert resp.status_code == 404


def _frame_extent() -> list[float]:
    f = _frame()
    return [f.roi.xmin, f.roi.xmax, f.roi.ymin, f.roi.ymax,
            f.depth_range.zmin, f.depth_range.zmax]
