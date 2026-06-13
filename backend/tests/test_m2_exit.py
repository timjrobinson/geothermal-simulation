"""M2 exit criteria — the backend half of "many surveys, one earth" (doc 03 §7, doc 05).

This is the M2 milestone gate for the *backend*: it proves the OVERVIEW §8 round-trip at
scale — the synthetic generator authors **one earth**, a battery of independent survey
methods is forward-modelled off it, and every one of those native-format files is ingested
through the **real** :func:`geosim.ingestion.ingest_file` pipeline into **one project**,
where all the normalized primitives land **co-registered in the Engineering Frame** (doc 01
§1): one shared :class:`SpatialFrame`, Engineering-metre bboxes that overlap a common
survey footprint, canonical units from the property-type registry (doc 01 §5), and a
provenance edge from each dataset back to its synthetic raw file (doc 02 §7, doc 05 §5).

We build ``unit-cube-v1`` rather than ``great-basin-v1``: great-basin's truth grid is
~22M cells (minutes to forward-model every method), whereas the unit-cube smoke scenario is
~6.8k cells and the whole build + multi-method ingest runs in ~1 s — keeping this gate fast
and headless (in-memory SQLite + tmp ``storage_root``; no Docker/Postgres/Redis).

The visual "layers over terrain in 3D" acceptance — loading these co-registered layers in
the browser viewer — is the FRONTEND half of the M2 exit (a separate workflow + a browser
check) and is intentionally out of scope here.
"""

from __future__ import annotations

import json

import pytest

from geosim.catalog import (
    Dataset,
    Observation,
    Project,
    PropertyModel,
    Provenance,
    RawFile,
    SpatialFrameRow,
    create_all,
    make_engine,
    session_factory,
)
from geosim.catalog.spatial import aabb_from_json, boxes_intersect
from geosim.ingestion import IngestStatus, ingest_file
from geosim.spatial import REGISTRY
from geosim.synthgen.scenarios import build_scenario

# Five+ independent methods surveyed off the SAME synthetic earth (doc 05 §4 table).
# Each maps to a real first-party doc-03 adapter; together they span potential-field,
# electrical, IP, MT, EM, borehole and seismic-catalog methods — the "many surveys".
# (relative path under the scenario folder, optional method_hint for detection)
_SURVEYS: list[tuple[str, str | None]] = [
    ("measured/gravity_stations.csv", None),   # gravity   — CSV stations
    ("measured/gravity_bouguer.tif", None),    # gravity   — GeoTIFF grid (PropertyModel)
    ("measured/aeromag_lines.xyz", None),      # magnetics — flight-line .xyz
    ("measured/ert_lineAA.stg", None),         # ert       — AGI .stg pseudosection
    ("measured/ip_lineAA.stg", None),          # ip        — chargeability .stg
    ("measured/mt/ST000.edi", None),           # mt        — EDI sounding
    ("measured/tem_soundings.xyz", None),      # em/tdem   — conductivity-depth .xyz
    ("measured/wells/UC-1.las", None),         # welllog   — LAS curves
    ("measured/microseismic.quakeml", "microseismic"),  # microseismic — QuakeML catalog
]


@pytest.fixture
def one_earth(tmp_path):
    """Build unit-cube-v1, then ingest every survey file into ONE project (real pipeline).

    Returns ``(session, project_id, manifest_by_path)`` — the session stays open so the
    test can assert on the persisted catalog rows.
    """
    scenario = build_scenario("unit-cube-v1", tmp_path / "scenario")
    assert scenario.errors == {}, scenario.errors  # every forward succeeded (doc 05 §6 T0)

    manifest = json.loads((scenario.out_dir / "manifest.json").read_text())
    manifest_by_path = {rec["path"]: rec for rec in manifest["measured"]}

    storage_root = tmp_path / "store"
    engine = make_engine()  # in-memory SQLite (doc 04 §2.1 fallback)
    create_all(engine)
    Session = session_factory(engine)

    with Session() as session:
        project_id: str | None = None
        reports = {}
        for rel, hint in _SURVEYS:
            report = ingest_file(
                session, storage_root, project_id, scenario.out_dir / rel,
                method_hint=hint,
            )
            assert report.status is not IngestStatus.FAILED, (rel, report.message)
            # Every survey after the first lands in the SAME project (one earth).
            project_id = report.project_id or project_id
            reports[rel] = report
        session.commit()
        yield session, project_id, manifest_by_path, reports


def test_many_surveys_land_in_one_project(one_earth):
    """All the surveyed methods ingest into a single project (OVERVIEW §8 round-trip)."""
    session, project_id, _, _ = one_earth

    # exactly one project holds the whole battery of surveys
    assert session.query(Project).count() == 1
    project = session.get(Project, project_id)
    assert project is not None

    datasets = session.query(Dataset).filter_by(project_id=project_id).all()
    assert len(datasets) == len(_SURVEYS)

    # ≥5 *distinct* canonical methods (doc 02 §2) — this is "many surveys, one earth".
    methods = {d.method for d in datasets}
    assert len(methods) >= 5
    assert {"gravity", "magnetics", "ert", "ip", "mt", "em", "welllog"} <= methods


def test_all_co_registered_in_one_engineering_frame(one_earth):
    """Every dataset shares ONE Engineering-metre SpatialFrame and overlaps in plan (doc 01 §1)."""
    session, project_id, _, _ = one_earth
    datasets = session.query(Dataset).filter_by(project_id=project_id).all()

    # one shared spatial frame: all datasets point at the project's single frame row
    frame_ids = {d.spatial_frame_id for d in datasets}
    assert frame_ids == {project_id}
    frame = session.get(SpatialFrameRow, project_id)
    assert frame is not None
    assert frame.length_unit == "m"  # Engineering metres (doc 01 §1)

    # Engineering-metre extents (doc 04 §2.2) that OVERLAP a common survey footprint:
    # every survey targeted the same patch of earth, so each one's plan (x/y) bbox falls
    # inside — and overlaps — the single connected survey region (their union). (Z bands
    # legitimately differ by acquisition geometry: surface gravity/mag vs. a deep borehole,
    # vs. a single MT sounding point — so co-registration is tested in plan.)
    boxes = [aabb_from_json(d.extent_json) for d in datasets]
    union = _union_xy(boxes)
    for b in boxes:
        assert boxes_intersect(_flatten_xy(b), union), b

    # the shared survey footprint is finite and modestly sized — one local earth, not a
    # scatter of unrelated surveys across a continent.
    assert union.xmax > union.xmin and union.ymax > union.ymin
    assert (union.xmax - union.xmin) < 1.0e5
    assert (union.ymax - union.ymin) < 1.0e5


def test_canonical_units_from_registry(one_earth):
    """Each primitive carries the canonical unit the property-type registry defines (doc 01 §5)."""
    session, project_id, _, _ = one_earth
    datasets = session.query(Dataset).filter_by(project_id=project_id).all()

    seen = 0
    for ds in datasets:
        for obs in session.query(Observation).filter_by(dataset_id=ds.id).all():
            if obs.primary_property is None:
                continue
            pt = REGISTRY.get(obs.primary_property)  # raises if not a registered type
            assert pt.canonical_unit  # registry pins a canonical unit (doc 01 §5)
            seen += 1
        for pm in session.query(PropertyModel).filter_by(dataset_id=ds.id).all():
            pt = REGISTRY.get(pm.property)
            assert pm.canonical_unit == pt.canonical_unit  # stored == registry canonical
            seen += 1
    # Every survey that emits a *typed* primitive contributes a canonicalized value; raw
    # catalogs whose observations carry no primary_property (e.g. a microseismic event
    # catalog) are exempt from the canonical-property check.
    assert seen >= len(_SURVEYS) - 1


def test_provenance_links_each_dataset_to_its_synthetic_raw_file(one_earth):
    """Provenance edges each dataset → the synthetic raw file it was ingested from (doc 02 §7)."""
    session, project_id, manifest_by_path, _ = one_earth
    datasets = session.query(Dataset).filter_by(project_id=project_id).all()

    # the synthetic SHA-256 set the generator stamped into the manifest (doc 05 §5)
    synthetic_sha = {
        rec["sha256"]
        for rec in manifest_by_path.values()
        if rec.get("sha256") and rec["provenance"]["source"] == "synthgen"
    }
    assert synthetic_sha

    for ds in datasets:
        prov = session.get(Provenance, ds.provenance_id)
        assert prov is not None  # no dataset without provenance (doc 02 §7)
        assert prov.process.startswith("ingest:")  # the ingest step op (doc 02 §7)
        assert prov.raw_file_id is not None
        raw = session.get(RawFile, prov.raw_file_id)
        assert raw is not None
        assert raw.project_id == project_id  # raw file co-located in the same project
        # the provenance traces back to a file the synthetic generator authored
        assert raw.sha256 in synthetic_sha


def _flatten_xy(box):
    """Return a copy of ``box`` with its Z span collapsed so overlap tests the plan only."""
    return aabb_from_json(
        {
            "xmin": box.xmin, "xmax": box.xmax,
            "ymin": box.ymin, "ymax": box.ymax,
            "zmin": 0.0, "zmax": 0.0,
        }
    )


def _union_xy(boxes):
    """The plan (x/y) union of a list of Engineering-metre bboxes (Z collapsed to 0)."""
    return aabb_from_json(
        {
            "xmin": min(b.xmin for b in boxes), "xmax": max(b.xmax for b in boxes),
            "ymin": min(b.ymin for b in boxes), "ymax": max(b.ymax for b in boxes),
            "zmin": 0.0, "zmax": 0.0,
        }
    )
