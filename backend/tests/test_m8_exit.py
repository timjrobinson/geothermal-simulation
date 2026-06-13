"""M8 exit criteria — implicit geological model (OVERVIEW §6 L4 / doc 07; M8).

The M8 gate: build an IMPLICIT geomodel with GemPy (numpy backend), constrained by
doc-02 horizon/fault contacts + well formation tops, over the project ``SpatialFrame``
ROI × depthRange (Engineering Frame, Z-up, doc 01 §1), and expose it two ways (doc 02):

  (a) one **unitSolid** ``GeologicalFeature`` per stratigraphic unit, packed as a valid
      ``.glb`` (doc 02 §5) — these render through the existing M6 glTF ``FeatureLayer``;
  (b) a categorical ``lithology_class`` **PropertyModel** with a class-probability axis
      (or hard labels + a categories table) (doc 02 §10.2);

both catalogued with provenance (doc 02 §7 — nothing exists without provenance), and the
recovered stratigraphic **ORDER** / a known **contact depth** reproduced (loose check vs
the input constraints).

This gate is deliberately end-to-end but coarse/fast (a ≤14³ grid + a handful of
interface/orientation points, CLAUDE.md) so the numpy-backend interpolation runs in
seconds. It sits BESIDE the M7 fusion pipeline, not on its critical path (doc 07 §6).

The input is a SMALL hand-authored layered+faulted setup consistent with a great-basin
stratigraphy: two flat-lying horizons stacked top→bottom over an implicit basement, cut
by one steeply-dipping N–S fault — the canonical Basin-&-Range structural motif. A couple
of the deeper-horizon interface points are supplied as **well formation tops** (doc 02 §5)
to exercise that path too.
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
    Dataset,
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

_GLB_MAGIC = 0x46546C67  # "glTF" little-endian

# The ROI × depthRange the model is built over (Engineering m, Z-up).
_ROI = Aabb(0.0, 1000.0, 0.0, 1000.0)
_DEPTH = DepthRange(0.0, 500.0)

# Truth contacts the input constraints encode (the two horizon tops, Engineering z).
_LAYER1_TOP_Z = 300.0  # shallower horizon (layer1 over layer2)
_LAYER2_TOP_Z = 150.0  # deeper horizon (layer2 over basement)
_RESOLUTION = (14, 14, 14)


def _frame() -> SpatialFrame:
    return SpatialFrame(roi=_ROI, depth_range=_DEPTH)


def _frame_extent() -> list[float]:
    return [_ROI.xmin, _ROI.xmax, _ROI.ymin, _ROI.ymax, _DEPTH.zmin, _DEPTH.zmax]


def _surfaces() -> list[dict]:
    """Two flat horizons (top→bottom) + one steep N–S fault — a Basin-&-Range motif.

    Only three of each horizon's five samples are given as SURFACE points; the other two
    arrive as well formation tops (see :func:`_well_tops`) so the well-path interface
    contribution is exercised (doc 02 §5).
    """
    return [
        {"name": "layer1", "kind": "horizon",
         "points": [[100, 100, _LAYER1_TOP_Z], [500, 500, _LAYER1_TOP_Z],
                    [900, 900, _LAYER1_TOP_Z]]},
        {"name": "layer2", "kind": "horizon",
         "points": [[100, 100, _LAYER2_TOP_Z], [500, 500, _LAYER2_TOP_Z],
                    [900, 900, _LAYER2_TOP_Z]]},
        {"name": "fault1", "kind": "fault",
         "points": [[500, 100, 100], [500, 500, 250], [500, 900, 400],
                    [520, 300, 150], [520, 700, 350]]},
    ]


def _well_tops() -> list[dict]:
    """Formation tops picked along two off-line wells, on the two horizon contacts (§5)."""
    return [
        {"surface": "layer1", "x": 100, "y": 900, "z": _LAYER1_TOP_Z},
        {"surface": "layer1", "x": 900, "y": 100, "z": _LAYER1_TOP_Z},
        {"surface": "layer2", "x": 100, "y": 900, "z": _LAYER2_TOP_Z},
        {"surface": "layer2", "x": 900, "y": 100, "z": _LAYER2_TOP_Z},
    ]


@pytest.fixture(scope="module")
def built() -> GeoModelResult:
    """Build the small layered+faulted implicit model once (the GemPy compute is the slow bit)."""
    spec = spec_from_catalog_surfaces(
        _frame(), _surfaces(), well_tops=_well_tops(), resolution=_RESOLUTION
    )
    return build_geomodel(spec)


# ─────────────── M8 (core): an implicit GemPy model yields a per-cell lithology ───────────────
def test_implicit_model_yields_per_cell_lithology(built: GeoModelResult) -> None:
    """The GemPy implicit model produces a per-cell lithology over the regular grid (doc 07).

    In doc-02 ``[z, y, x]`` Z-up order, sized to the requested coarse resolution, with every
    label decoding to a row in the categories table.
    """
    nx, ny, nz = _RESOLUTION
    assert built.lith_zyx.shape == (nz, ny, nx)

    labels = np.unique(np.round(built.lith_zyx)).astype(int)
    # The two stratigraphic units + the implicit basement are all present (a layered earth).
    assert labels.size >= 3
    cat_ids = {c["id"] for c in built.categories}
    assert set(labels.tolist()).issubset(cat_ids)


def test_recovered_stratigraphic_order_and_contacts(built: GeoModelResult) -> None:
    """THE M8 EXIT (structure): the recovered stratigraphic ORDER + contacts match the input.

    The implicit field must reproduce the supplied top→bottom stack (layer1 over layer2 over
    the implicit basement) and place each horizon contact within ~one cell of the input
    constraint depth — a loose check that the interpolation honoured the doc-02 contacts.
    """
    labels = np.round(built.lith_zyx).astype(int)
    z0, _, _ = built.origin_zyx
    dz, _, _ = built.spacing_zyx
    nz = labels.shape[0]
    z_centres = z0 + dz * np.arange(nz)  # z-index increases UPWARD (Z-up, doc 01 §1)

    id_by_name = {c["name"]: c["id"] for c in built.categories}
    assert {"layer1", "layer2", "basement"} <= set(id_by_name)
    l1, l2, base = id_by_name["layer1"], id_by_name["layer2"], id_by_name["basement"]

    # Sample an interior column away from the fault to read the undisturbed stratigraphy.
    col = labels[:, labels.shape[1] // 4, labels.shape[2] // 4]
    z_l1 = z_centres[col == l1]
    z_l2 = z_centres[col == l2]
    z_base = z_centres[col == base]
    assert z_l1.size and z_l2.size and z_base.size, "all three units present in the column"

    # ORDER (Z-up): layer1 sits ABOVE layer2 sits ABOVE basement (top→bottom stack recovered).
    assert z_l1.min() > z_l2.max()
    assert z_l2.min() > z_base.max()

    # CONTACT depths reproduced within ~one cell of the input horizon tops (loose check).
    tol = 1.5 * dz
    recovered_l1_top = 0.5 * (z_l1.min() + z_l2.max())  # layer1/layer2 contact
    recovered_l2_top = 0.5 * (z_l2.min() + z_base.max())  # layer2/basement contact
    assert recovered_l1_top == pytest.approx(_LAYER1_TOP_Z, abs=tol)
    assert recovered_l2_top == pytest.approx(_LAYER2_TOP_Z, abs=tol)


def test_class_probability_axis_is_a_distribution(built: GeoModelResult) -> None:
    """Deliverable (b): a class-probability axis normalised across classes (doc 02 §10.2)."""
    n_class = built.class_prob.shape[0]
    strat_cats = [c for c in built.categories if not c["isFault"]]
    assert n_class == len(strat_cats)
    # (n_class, nz, ny, nx) aligned to the label volume.
    assert built.class_prob.shape[1:] == built.lith_zyx.shape
    # A proper distribution over the class axis: non-negative, sums to 1 per cell.
    assert built.class_prob.min() >= 0.0
    assert np.allclose(built.class_prob.sum(axis=0), 1.0, atol=1e-4)


def test_fault_is_modelled_and_flagged(built: GeoModelResult) -> None:
    """The N–S fault is carried as its own contact + flagged isFault (doc 02 §5)."""
    faults = [c for c in built.categories if c["isFault"]]
    assert any(c["name"] == "fault1" for c in faults)
    # The fault is NOT a stratigraphic class in the probability axis (only the strat units are).
    assert "fault1" not in {c["name"] for c in built.categories if not c["isFault"]}


# ─────── M8 deliverables (a)+(b): catalogued with provenance (doc 02 §5/§7/§10.2) ───────
@pytest.fixture
def env(tmp_path):
    """An in-memory catalog + a project layout under ``tmp_path`` (doc 04 §2.1 fallback)."""
    engine = make_engine()
    create_all(engine)
    session = session_factory(engine)()
    pid = new_id(IdKind.PROJECT)
    layout = ensure_project_layout(tmp_path, pid)
    session.add(Project(id=pid, name="m8-exit", storage_root=str(tmp_path)))
    session.add(SpatialFrameRow(project_id=pid, **frame_row_kwargs(_frame())))
    session.commit()
    yield session, layout, pid
    session.close()


def test_persists_lithology_model_unit_solids_and_provenance(
    env, built: GeoModelResult
) -> None:
    """THE M8 EXIT (products): a lithology_class PropertyModel + valid unitSolid glbs + provenance.

    Persisting the result writes (a) one ``unitSolid`` feature per stratigraphic unit as a
    structurally-valid ``.glb`` and (b) a categorical ``lithology_class`` PropertyModel with a
    categories table + a class-probability summary — both threaded onto a single
    ``model:gempy-implicit`` provenance row (doc 02 §5/§7/§10.2; doc 04 §2.4).
    """
    session, layout, pid = env
    persisted = persist_geomodel(
        session, layout, pid, built, extent=_frame_extent(), created_by="m8@test"
    )

    # --- provenance lineage (doc 02 §7) ---
    prov = session.get(Provenance, persisted.provenance_id)
    assert prov is not None and prov.process == "model:gempy-implicit"
    ds = session.get(Dataset, persisted.dataset_id)
    assert ds is not None and ds.kind == "propertyModel" and ds.provenance_id == prov.id

    # --- (b) categorical lithology_class PropertyModel (doc 02 §10.2) ---
    pm = session.get(PropertyModel, persisted.property_model_id)
    assert pm is not None
    assert pm.property == "lithology_class"
    assert pm.support == "volume"
    assert json.loads(pm.shape_json) == list(built.shape_zyx)

    reader = open_property_model(persisted.lithology_store_uri)
    assert "lithology_class" in reader.properties
    categories = reader.attrs("lithology_class")["categories"]
    assert categories and all({"index", "id", "name"} <= set(c) for c in categories)

    # Hard labels round-trip as integer-valued floats over the grid.
    level0 = reader.read_level("lithology_class")
    assert level0.shape == built.shape_zyx
    assert np.allclose(level0, np.round(level0))

    # The class-probability axis is recorded alongside the hard labels (doc 02 §10.2).
    cp = reader.group["lithology_class"].attrs["classProbability"]
    assert cp["axis"] == "class"
    assert cp["nClasses"] == built.class_prob.shape[0]

    # --- (a) one valid unitSolid feature per stratigraphic unit (doc 02 §5) ---
    assert persisted.unit_solid_feature_ids
    feats = session.query(Feature).filter_by(project_id=pid).all()
    assert feats and all(f.feature_type == "unitSolid" for f in feats)
    unit_names = {json.loads(f.props_json)["unit"] for f in feats}
    # The stratigraphic units present in the block each got a solid.
    present = {c["name"] for c in built.categories if not c["isFault"]
               and (np.round(built.lith_zyx).astype(int) == c["id"]).any()}
    assert unit_names == present

    for feat in feats:
        glb = open(feat.store_uri, "rb").read()
        magic, version, total = struct.unpack("<III", glb[:12])
        assert magic == _GLB_MAGIC  # a glTF binary container
        assert version == 2
        assert total == len(glb)  # the header's declared length matches the file (valid glb)
        props = json.loads(feat.props_json)
        assert props["triangleCount"] > 0 and props["vertexCount"] > 0
        assert props["source"] == "model:gempy-implicit"


# ─────────────── M8 deliverable: the HTTP build surface (doc 04 §9) ───────────────
def test_post_geomodel_endpoint_builds_and_catalogs(tmp_path) -> None:
    """``POST /projects/{pid}/geomodel`` builds the implicit model + returns the catalog ids.

    The same coarse layered+faulted spec, driven through the API: the response carries the
    lithology-model id, the unitSolid feature ids, the categories (fault flagged) and the
    grid shape, and the catalogued PropertyModel is fetchable via the M1 read surface.
    """
    app = create_app(Settings(storage_root=tmp_path))
    client = TestClient(app)
    pid = client.post(
        "/projects",
        json={"name": "m8-api", "frame": {
            "mode": "local",
            "roi": {"xmin": 0, "xmax": 1000, "ymin": 0, "ymax": 1000},
            "depth_range": {"zmin": 0, "zmax": 500},
        }},
    ).json()["id"]

    resp = client.post(
        f"/projects/{pid}/geomodel",
        json={
            "surfaces": _surfaces(),
            "wellTops": _well_tops(),
            "resolution": [12, 12, 12],
        },
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["propertyModelId"].startswith("pm_")
    assert body["unitSolidFeatureIds"]
    assert body["shape"] == [12, 12, 12]
    # The fault contact is catalogued + flagged; the strat stack is present.
    assert any(c["isFault"] and c["name"] == "fault1" for c in body["categories"])
    assert {"layer1", "layer2", "basement"} <= {c["name"] for c in body["categories"]}

    # The lithology_class PropertyModel is fetchable via the M1 read surface (doc 04 §9.2).
    meta = client.get(f"/property-models/{body['propertyModelId']}")
    assert meta.status_code == 200
    assert meta.json()["property"] == "lithology_class"


def test_post_geomodel_404_unknown_project(tmp_path) -> None:
    """An unknown project id is a 404 (doc 04 §9 error contract)."""
    app = create_app(Settings(storage_root=tmp_path))
    client = TestClient(app)
    resp = client.post(
        "/projects/prj_does_not_exist/geomodel", json={"surfaces": _surfaces()}
    )
    assert resp.status_code == 404
