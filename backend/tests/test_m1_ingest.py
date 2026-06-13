"""M1 synthgen + ingest 'write + register' tests (doc 03 §7, doc 05, doc 02 §6/§7/§10).

In-memory SQLite (doc 04 §2.1 fallback) + a tmp ``storage_root`` — no Docker/Postgres.
Asserts the deterministic synthetic volume, the doc-02 Zarr layout on disk, and the
catalog rows + provenance edge written by :func:`geosim.ingestion.seed_m1_project`.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from geosim.catalog import (
    Dataset,
    PropertyModel,
    Provenance,
    SpatialFrameRow,
    create_all,
    is_kind,
    make_engine,
    session_factory,
)
from geosim.catalog.ids import IdKind
from geosim.ingestion import seed_m1_project
from geosim.storage import SIGMA_SUFFIX, open_property_model
from geosim.synthgen import build_resistivity_volume


@pytest.fixture
def session():
    engine = make_engine()  # in-memory SQLite (doc 04 §2.1)
    create_all(engine)
    Session = session_factory(engine)
    with Session() as s:
        yield s


# ─────────────────────────── synthgen (doc 05) ───────────────────────────
def test_volume_is_deterministic():
    a = build_resistivity_volume(shape=(16, 16, 16), seed=7)
    b = build_resistivity_volume(shape=(16, 16, 16), seed=7)
    assert np.array_equal(a.values, b.values)
    assert np.array_equal(a.sigma, b.sigma)
    # different seed → different realization
    c = build_resistivity_volume(shape=(16, 16, 16), seed=8)
    assert not np.array_equal(a.values, c.values)


def test_volume_physically_plausible_and_float32():
    v = build_resistivity_volume(shape=(24, 24, 24), seed=1)
    assert v.values.dtype == np.float32
    assert v.sigma.dtype == np.float32
    assert v.values.shape == (24, 24, 24)
    assert np.all(np.isfinite(v.values))
    # background in a plausible band (doc 05 §2.2): 100–500 Ω·m order, blob pulls a low
    assert v.values.max() <= 600.0
    assert v.values.min() <= 20.0  # the conductive blob core (5–20 Ω·m)
    assert np.all(v.sigma > 0)


def test_blob_is_more_conductive_than_background():
    v = build_resistivity_volume(shape=(32, 32, 32), seed=42)
    cz, cy, cx = (int(round(c)) for c in v.blob_center)
    blob_val = v.values[cz, cy, cx]
    # a corner is well outside the blob → background
    bg_val = v.values[0, 0, 0]
    assert blob_val < bg_val
    assert blob_val < 20.0  # conductive geothermal anomaly band


# ─────────────────────────── ingest: write + register (doc 03 §7) ───────────────────────────
def test_seed_m1_project_registers_property_model(session, tmp_path):
    storage_root = tmp_path / "storage"
    ids = seed_m1_project(session, storage_root, shape=(32, 32, 32), seed=42)

    assert is_kind(ids["project_id"], IdKind.PROJECT)
    assert is_kind(ids["dataset_id"], IdKind.DATASET)
    assert is_kind(ids["property_model_id"], IdKind.PROPERTY_MODEL)

    # ── property_models catalog row: property / unit / shape / levels ──
    pm = session.get(PropertyModel, ids["property_model_id"])
    assert pm is not None
    assert pm.property == "resistivity"
    assert pm.canonical_unit == "ohm*m"
    assert pm.support == "volume"
    assert pm.store_format == "zarr"
    assert json.loads(pm.shape_json) == [32, 32, 32]
    assert pm.pyramid_levels >= 1
    assert pm.uncertainty_uri == f"resistivity{SIGMA_SUFFIX}"
    assert pm.dataset_id == ids["dataset_id"]
    assert pm.project_id == ids["project_id"]

    # ── dataset envelope (doc 02 §2) ──
    ds = session.get(Dataset, ids["dataset_id"])
    assert ds.kind == "propertyModel"
    assert ds.status == "ready"
    assert ds.provenance_id is not None

    # ── spatial frame is local-mode (doc 01 §2) ──
    frame_row = session.get(SpatialFrameRow, ids["project_id"])
    assert frame_row.mode == "local"


def test_seed_m1_writes_doc02_zarr_layout(session, tmp_path):
    storage_root = tmp_path / "storage"
    ids = seed_m1_project(session, storage_root, shape=(32, 32, 32), seed=42)

    pm = session.get(PropertyModel, ids["property_model_id"])
    zarr_path = pm.store_uri

    reader = open_property_model(zarr_path)
    assert "resistivity" in reader.properties
    # <property>/0 full-resolution array exists and round-trips shape
    level0 = reader.read_level("resistivity", 0)
    assert level0.shape == (32, 32, 32)
    # sibling _sigma subgroup (doc 02 §6) with the same shape
    assert reader.has_sigma("resistivity")
    sigma0 = reader.read_sigma_level("resistivity", 0)
    assert sigma0.shape == level0.shape
    # OME-Zarr multiscales present, levels match the catalog (doc 02 §10.3)
    assert reader.level_count("resistivity") == pm.pyramid_levels
    assert reader.multiscales("resistivity")[0]["datasets"]
    # origin/spacing attrs (z,y,x) recorded (doc 02 §10.2)
    attrs = reader.attrs("resistivity", 0)
    assert attrs["origin"] == [0.0, 0.0, 0.0]
    assert attrs["spacing"] == [25.0, 25.0, 25.0]


def test_seed_m1_links_provenance(session, tmp_path):
    storage_root = tmp_path / "storage"
    ids = seed_m1_project(session, storage_root, seed=42)

    ds = session.get(Dataset, ids["dataset_id"])
    prov = session.get(Provenance, ds.provenance_id)
    assert prov is not None
    assert prov.process == "synthesize"  # doc 02 §7 Step op
    assert prov.target_kind == "propertyModel"
    assert prov.target_id == ids["property_model_id"]
    assert prov.project_id == ids["project_id"]
    params = json.loads(prov.params_json)
    assert params["agent"] == "geosim.ingestion.seed_m1_project"
    assert "synthgen" in params["tool"]
    assert params["params"]["seed"] == 42


def test_seed_m1_blob_more_conductive_in_store(session, tmp_path):
    """The doc-05 conductive anomaly survives the write: blob < background on disk."""
    storage_root = tmp_path / "storage"
    ids = seed_m1_project(session, storage_root, shape=(32, 32, 32), seed=42)
    pm = session.get(PropertyModel, ids["property_model_id"])

    reader = open_property_model(pm.store_uri)
    vol = reader.read_level("resistivity", 0)
    nz, ny, nx = vol.shape
    blob_val = vol[nz // 2, ny // 2, nx // 2]
    bg_val = vol[0, 0, 0]
    assert blob_val < bg_val
    assert blob_val < 20.0
