"""Tests for the serving API (doc 04 §9, doc 08 §7.1).

All tests run on the no-service embedded tier (SQLite in-memory + temp ``storage_root`` +
:class:`InlineJobRunner`) — no Docker/Redis/Postgres (HARD RULE). They exercise the
doc-04 §9.2 contract: project CRUD creates BOTH a catalog row AND the on-disk project
directory; ``GET /api/capabilities`` returns the seeded doc-01 §5 property types; and the
demo job reaches ``progress=1.0`` / ``status=succeeded`` (doc 04 §9.4).
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from geosim.api import Settings, create_app
from geosim.catalog import Project as ProjectRow
from geosim.catalog import SpatialFrameRow


@pytest.fixture
def settings(tmp_path):
    """A no-service Settings: SQLite in-memory + temp storage root + inline runner."""
    return Settings(storage_root=tmp_path / "storage")


@pytest.fixture
def client(settings):
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def session_factory(client):
    return client.app.state.session_factory


# ──────────────────────────── capabilities (doc 08 §7.1) ────────────────────────────
def test_capabilities_shape_and_seeded_property_types(client):
    resp = client.get("/api/capabilities")
    assert resp.status_code == 200
    doc = resp.json()
    # The single backend→frontend contract keys (doc 08 §7.1).
    for key in ("api_version", "property_types", "methods", "renderers", "transforms", "plugins"):
        assert key in doc
    # Property types flow straight from the doc-01 §5 registry — present with zero plugins.
    keys = {pt["key"] for pt in doc["property_types"]}
    assert "resistivity" in keys
    assert "density" in keys
    res = next(pt for pt in doc["property_types"] if pt["key"] == "resistivity")
    assert {"key", "unit", "colormap", "scaling"} <= set(res)
    assert res["scaling"] == "log"


def test_capabilities_matches_frontend_store_shape(client):
    """The frontend store.ts `Capabilities` interface keys must all be emitted."""
    doc = client.get("/api/capabilities").json()
    assert isinstance(doc["api_version"], str)
    for pt in doc["property_types"]:
        assert {"key", "unit", "colormap", "scaling"} <= set(pt)
    for m in doc["methods"]:
        assert {"id", "name"} <= set(m)
    for p in doc["plugins"]:
        assert {"id", "version"} <= set(p)


# ──────────────────────────── projects (doc 04 §9.2) ────────────────────────────
def test_create_project_writes_catalog_row_and_directory(client, settings, session_factory):
    resp = client.post("/projects", json={"name": "Hot Field"})
    assert resp.status_code == 201
    proj = resp.json()
    pid = proj["id"]
    assert pid.startswith("prj_")
    assert proj["name"] == "Hot Field"
    assert proj["frame"]["mode"] == "local"  # default doc-01 frame

    # 1) catalog row exists.
    with session_factory() as s:
        row = s.get(ProjectRow, pid)
        assert row is not None
        assert row.name == "Hot Field"
        frame_row = s.execute(
            select(SpatialFrameRow).where(SpatialFrameRow.project_id == pid)
        ).scalar_one()
        assert frame_row.mode == "local"

    # 2) project directory + bulk-store tree on disk (doc 04 §3).
    proj_dir = settings.resolved_storage_root() / pid
    assert proj_dir.is_dir()
    for store in ("arrays", "grids", "meshes", "vectors", "points", "raw", "cache"):
        assert (proj_dir / store).is_dir()
    assert (proj_dir / "frame.json").is_file()


def test_create_project_with_georeferenced_frame(client):
    frame = {
        "mode": "georeferenced",
        "horizontal_crs": "EPSG:32612",
        "anchor": {"easting": 500000.0, "northing": 4000000.0, "elevation": 1200.0},
        "rotation_deg": 0.0,
        "roi": {"xmin": -1000, "xmax": 1000, "ymin": -1000, "ymax": 1000},
        "depth_range": {"zmin": -3000, "zmax": 500},
    }
    resp = client.post("/projects", json={"name": "Geo", "frame": frame})
    assert resp.status_code == 201
    got = resp.json()["frame"]
    assert got["mode"] == "georeferenced"
    assert got["horizontal_crs"] == "EPSG:32612"
    assert got["anchor"]["easting"] == 500000.0


def test_list_get_patch_delete_project(client):
    pid = client.post("/projects", json={"name": "A"}).json()["id"]

    listed = client.get("/projects").json()
    assert any(p["id"] == pid for p in listed)

    got = client.get(f"/projects/{pid}").json()
    assert got["name"] == "A"
    assert "frame" in got and "storage_root" in got

    patched = client.patch(f"/projects/{pid}", json={"name": "B"}).json()
    assert patched["name"] == "B"

    # frame patch = doc-01 georeference promote
    patched2 = client.patch(
        f"/projects/{pid}",
        json={"frame": {"mode": "georeferenced", "horizontal_crs": "EPSG:32601",
                        "anchor": {"easting": 1.0, "northing": 2.0, "elevation": 3.0}}},
    ).json()
    assert patched2["frame"]["mode"] == "georeferenced"

    assert client.delete(f"/projects/{pid}").status_code == 204
    assert client.get(f"/projects/{pid}").status_code == 404


def test_get_missing_project_404(client):
    assert client.get("/projects/prj_nope").status_code == 404


# ──────────────────────────── jobs (doc 04 §9.2, §9.4) ────────────────────────────
def test_demo_job_succeeds_and_reaches_progress_one(client, session_factory):
    pid = client.post("/projects", json={"name": "Jobs"}).json()["id"]

    enq = client.post(f"/projects/{pid}/jobs:demo", json={"steps": 4})
    assert enq.status_code == 202
    jid = enq.json()["job_id"]
    assert jid.startswith("job_")

    job = client.get(f"/jobs/{jid}").json()
    assert job["status"] == "succeeded"
    assert job["progress"] == 1.0
    assert job["result"] == {"steps": 4, "done": True}

    # Durable jobs row persisted (doc 04 §2.4 — source of truth across reload).
    from geosim.catalog import Job as JobRow

    with session_factory() as s:
        row = s.get(JobRow, jid)
        assert row is not None
        assert row.status == "succeeded"
        assert row.progress == 1.0
        assert json.loads(row.result_json) == {"steps": 4, "done": True}


def test_demo_job_default_steps(client):
    pid = client.post("/projects", json={"name": "J2"}).json()["id"]
    jid = client.post(f"/projects/{pid}/jobs:demo").json()["job_id"]
    job = client.get(f"/jobs/{jid}").json()
    assert job["status"] == "succeeded"
    assert job["progress"] == 1.0


def test_demo_job_on_missing_project_404(client):
    assert client.post("/projects/prj_nope/jobs:demo").status_code == 404


def test_get_missing_job_404(client):
    assert client.get("/jobs/job_nope").status_code == 404


def test_ws_progress_streams_to_terminal(client):
    """WS /jobs/{jid}/progress replays the buffered 0→1 progression + terminal event."""
    pid = client.post("/projects", json={"name": "WS"}).json()["id"]
    jid = client.post(f"/projects/{pid}/jobs:demo", json={"steps": 3}).json()["job_id"]

    events = []
    with client.websocket_connect(f"/jobs/{jid}/progress") as ws:
        try:
            while True:
                events.append(ws.receive_json())
        except Exception:
            pass

    assert events, "expected at least one progress event"
    last = events[-1]
    assert last["status"] == "succeeded"
    assert last["progress"] == 1.0


def test_ws_unknown_job_closes(client):
    with pytest.raises(Exception):
        with client.websocket_connect("/jobs/job_nope/progress") as ws:
            ws.receive_json()
