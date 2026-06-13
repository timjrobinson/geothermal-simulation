"""Tests for the catalog DB (doc 04 §2).

In-memory SQLite (the doc 04 §2.1 lightweight fallback), ``create_all`` bootstrap,
ULID prefix invariants (doc 02 §1), Engineering-metre bbox intersection (doc 04
§2.5), and FK cascade on project delete (doc 04 §2.4 ON DELETE CASCADE).
"""

import json

import pytest
from sqlalchemy import select

from geosim.catalog import (
    Bbox3D,
    Dataset,
    Feature,
    FusedLayer,
    FusedModel,
    IdKind,
    Observation,
    Project,
    Provenance,
    PropertyModel,
    SpatialFrameRow,
    create_all,
    is_kind,
    make_engine,
    new_id,
    prefix_of,
    query_artifacts_bbox,
    query_datasets_bbox,
    session_factory,
)
from geosim.spatial import Aabb, DepthRange, SpatialFrame


@pytest.fixture
def session():
    engine = make_engine()  # in-memory SQLite fallback (doc 04 §2.1)
    create_all(engine)
    Session = session_factory(engine)
    with Session() as s:
        yield s


def _aabb_json(xmin, xmax, ymin, ymax, zmin, zmax) -> str:
    return json.dumps(
        {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax,
         "zmin": zmin, "zmax": zmax}
    )


def _seed_project(s, *, project_id=None) -> tuple[str, str, str]:
    """Insert project + spatial_frame + provenance; return their ids."""
    pid = project_id or new_id(IdKind.PROJECT)
    s.add(Project(id=pid, name="Milford", storage_root=f"/data/{pid}"))

    frame = SpatialFrame(roi=Aabb(-5000, 5000, -5000, 5000),
                         depth_range=DepthRange(-8000, 2000))
    s.add(SpatialFrameRow(
        project_id=pid, mode=frame.mode.value,
        roi_json=json.dumps({"xmin": -5000, "xmax": 5000, "ymin": -5000, "ymax": 5000}),
        depth_range_json=json.dumps({"zmin": -8000, "zmax": 2000}),
        frame_json=json.dumps({"mode": frame.mode.value}),
    ))

    prov_id = new_id(IdKind.PROVENANCE)
    s.add(Provenance(id=prov_id, project_id=pid, target_kind="dataset",
                     target_id="(pending)", process="ingest:synthetic"))
    s.flush()
    return pid, pid, prov_id  # spatial_frame_id == project_id (1:1)


# ─────────────────────────── ULID prefixes (doc 02 §1) ───────────────────────────


def test_ulid_prefixes_correct():
    assert prefix_of(new_id(IdKind.DATASET)) == "ds"
    assert prefix_of(new_id(IdKind.PROPERTY_MODEL)) == "pm"
    assert prefix_of(new_id(IdKind.OBSERVATION)) == "obs"
    assert prefix_of(new_id(IdKind.FEATURE)) == "feat"
    assert prefix_of(new_id(IdKind.FUSED_MODEL)) == "fem"
    assert prefix_of(new_id(IdKind.PROVENANCE)) == "prov"
    assert prefix_of(new_id(IdKind.WELL)) == "well"
    assert prefix_of(new_id(IdKind.VERSION)) == "ver"
    assert prefix_of(new_id(IdKind.RUN)) == "run"
    assert prefix_of(new_id(IdKind.PROJECT)) == "prj"
    assert is_kind(new_id(IdKind.FUSED_MODEL), IdKind.FUSED_MODEL)
    # ULID body is 26 chars of Crockford base32 after the '<token>_' prefix.
    body = new_id(IdKind.DATASET).split("_", 1)[1]
    assert len(body) == 26


def test_ulids_sort_time_ordered():
    ids = [new_id(IdKind.DATASET) for _ in range(5)]
    assert ids == sorted(ids)


# ─────────────────────────── schema bootstrap + insert ───────────────────────────


def test_create_all_and_insert_core_rows(session):
    pid, frame_id, prov_id = _seed_project(session)

    ds_id = new_id(IdKind.DATASET)
    session.add(Dataset(
        id=ds_id, project_id=pid, name="MT inversion", method="mt", kind="propertyModel",
        status="ready", extent_json=_aabb_json(-1000, 1000, -1000, 1000, -2000, 0),
        spatial_frame_id=frame_id, provenance_id=prov_id,
        version_root_id=ds_id, version_seq=1, created_by="system:synthetic",
    ))

    pm_id = new_id(IdKind.PROPERTY_MODEL)
    session.add(PropertyModel(
        id=pm_id, dataset_id=ds_id, project_id=pid, property="resistivity",
        canonical_unit="ohm*m", support="volume",
        store_uri=f"arrays/{pm_id}.zarr", shape_json=json.dumps([64, 128, 128]),
        bbox_json=_aabb_json(-1000, 1000, -1000, 1000, -2000, 0),
    ))
    session.commit()

    got = session.get(PropertyModel, pm_id)
    assert got is not None
    assert got.canonical_unit == "ohm*m"
    assert got.dataset.method == "mt"
    assert json.loads(got.shape_json) == [64, 128, 128]


# ─────────────────────────── bbox intersection (doc 04 §2.5) ───────────────────────────


def test_bbox_intersection_returns_right_rows(session):
    pid, frame_id, prov_id = _seed_project(session)

    def make_dataset_with_pm(name, box):
        ds_id = new_id(IdKind.DATASET)
        session.add(Dataset(
            id=ds_id, project_id=pid, name=name, method="ert", kind="propertyModel",
            status="ready", extent_json=_aabb_json(*box), spatial_frame_id=frame_id,
            provenance_id=prov_id, version_root_id=ds_id, created_by="t@x",
        ))
        pm_id = new_id(IdKind.PROPERTY_MODEL)
        session.add(PropertyModel(
            id=pm_id, dataset_id=ds_id, project_id=pid, property="resistivity",
            canonical_unit="ohm*m", support="volume", store_uri=f"arrays/{pm_id}.zarr",
            shape_json=json.dumps([32, 32, 32]), bbox_json=_aabb_json(*box),
        ))
        return ds_id, pm_id

    # A: near origin (intersects query), B: far away (does not)
    a_ds, a_pm = make_dataset_with_pm("near", (-500, 500, -500, 500, -1000, 0))
    b_ds, b_pm = make_dataset_with_pm("far", (4000, 5000, 4000, 5000, -1000, 0))
    session.commit()

    query = Bbox3D(-100, 100, -100, 100, -800, 0)
    ds_hits = query_datasets_bbox(session, pid, query)
    assert {d.id for d in ds_hits} == {a_ds}

    art_hits = query_artifacts_bbox(session, pid, query, kinds=["propertyModel"])
    assert {a.id for a in art_hits} == {a_pm}

    # depth clipping: a query below the volumes prunes everything (doc 04 §2.5)
    deep = Bbox3D(-100, 100, -100, 100, -5000, -4000)
    assert query_datasets_bbox(session, pid, deep) == []


def test_bbox_touching_boundary_intersects(session):
    pid, frame_id, prov_id = _seed_project(session)
    ds_id = new_id(IdKind.DATASET)
    session.add(Dataset(
        id=ds_id, project_id=pid, name="edge", method="gravity", kind="propertyModel",
        status="ready", extent_json=_aabb_json(0, 100, 0, 100, -100, 0),
        spatial_frame_id=frame_id, provenance_id=prov_id, version_root_id=ds_id,
        created_by="t@x",
    ))
    session.commit()
    # query box shares the x=100 face exactly → counts as intersecting (inclusive)
    touching = Bbox3D(100, 200, 0, 100, -100, 0)
    assert {d.id for d in query_datasets_bbox(session, pid, touching)} == {ds_id}


# ─────────────────────────── FK cascade on project delete (doc 04 §2.4) ───────────────────────────


def test_fk_cascade_on_project_delete(session):
    pid, frame_id, prov_id = _seed_project(session)
    ds_id = new_id(IdKind.DATASET)
    session.add(Dataset(
        id=ds_id, project_id=pid, name="ds", method="seismic", kind="propertyModel",
        status="ready", extent_json=_aabb_json(-1, 1, -1, 1, -1, 0),
        spatial_frame_id=frame_id, provenance_id=prov_id, version_root_id=ds_id,
        created_by="t@x",
    ))
    pm_id = new_id(IdKind.PROPERTY_MODEL)
    session.add(PropertyModel(
        id=pm_id, dataset_id=ds_id, project_id=pid, property="velocity_p",
        canonical_unit="m/s", support="volume", store_uri=f"arrays/{pm_id}.zarr",
        shape_json=json.dumps([8, 8, 8]), bbox_json=_aabb_json(-1, 1, -1, 1, -1, 0),
    ))
    obs_id = new_id(IdKind.OBSERVATION)
    session.add(Observation(
        id=obs_id, dataset_id=ds_id, project_id=pid, geometry_kind="points",
        bbox_json=_aabb_json(-1, 1, -1, 1, -1, 0),
    ))
    session.commit()

    # delete via ORM so cascades on the relationships also flush the spatial_frame
    proj = session.get(Project, pid)
    session.delete(proj)
    session.commit()

    assert session.scalars(select(Dataset).where(Dataset.project_id == pid)).all() == []
    assert session.scalars(select(PropertyModel).where(PropertyModel.project_id == pid)).all() == []
    assert session.scalars(select(Observation).where(Observation.project_id == pid)).all() == []
    assert session.scalars(select(SpatialFrameRow).where(SpatialFrameRow.project_id == pid)).all() == []


def test_db_level_fk_cascade_via_sql(session):
    """SQLite ON DELETE CASCADE actually fires (PRAGMA foreign_keys=ON, doc 04 §2.4)."""
    pid, frame_id, prov_id = _seed_project(session)
    fem_ds = new_id(IdKind.DATASET)
    session.add(Dataset(
        id=fem_ds, project_id=pid, name="fused", method="fused", kind="fusedModel",
        status="ready", extent_json=_aabb_json(-1, 1, -1, 1, -1, 0),
        spatial_frame_id=frame_id, provenance_id=prov_id, version_root_id=fem_ds,
        created_by="system:fusion",
    ))
    fem_id = new_id(IdKind.FUSED_MODEL)
    session.add(FusedModel(
        id=fem_id, dataset_id=fem_ds, project_id=pid, store_uri=f"arrays/{fem_id}.zarr",
        shape_json=json.dumps([32, 32, 32]), spacing_json=json.dumps([10, 10, 10]),
        origin_json=json.dumps([0, 0, 0]), bbox_json=_aabb_json(-1, 1, -1, 1, -1, 0),
    ))
    pm_id = new_id(IdKind.PROPERTY_MODEL)
    session.add(PropertyModel(
        id=pm_id, dataset_id=fem_ds, project_id=pid, property="density",
        canonical_unit="kg/m**3", support="volume", store_uri=f"arrays/{pm_id}.zarr",
        shape_json=json.dumps([8, 8, 8]), bbox_json=_aabb_json(-1, 1, -1, 1, -1, 0),
    ))
    flay_id = new_id(IdKind.FUSED_LAYER)
    session.add(FusedLayer(
        id=flay_id, fused_model_id=fem_id, source_property_model_id=pm_id,
        source_version="ver_1", property="density",
        resample_op_json=json.dumps({"method": "trilinear"}),
    ))
    feat_id = new_id(IdKind.FEATURE)
    session.add(Feature(
        id=feat_id, dataset_id=fem_ds, project_id=pid, feature_type="fault",
        store_format="gltf", bbox_json=_aabb_json(-1, 1, -1, 1, -1, 0),
    ))
    session.commit()

    # delete the dataset directly → fused_models, fused_layers, features cascade in-DB
    session.execute(Dataset.__table__.delete().where(Dataset.id == fem_ds))
    session.commit()
    assert session.get(FusedModel, fem_id) is None
    assert session.get(FusedLayer, flay_id) is None  # cascaded via fused_models
    assert session.get(Feature, feat_id) is None
