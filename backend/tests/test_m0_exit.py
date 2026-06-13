"""End-to-end M0 exit-criterion integration test (ROADMAP.md "M0 — Foundations").

The M0 exit criterion is the walking skeleton, exercised here against the *real*
foundation packages (no Docker/Redis/Postgres, HARD RULE — SQLite in-memory + temp
``storage_root`` + :class:`~geosim.jobs.InlineJobRunner`):

    create a project via the API  →  a catalog row in the DB **and** a project directory
    on disk (doc 04 §3 bulk-store tree)  →  enqueue the trivial demo job  →  progress
    streams over the WS endpoint to completion (``status == "succeeded"``, doc 04 §9.4).

This single flow ties together :mod:`geosim.api`, :mod:`geosim.catalog`,
:mod:`geosim.storage`, and :mod:`geosim.jobs` — the storage→catalog→job→serving spine
the rest of the roadmap builds on.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from geosim.api import Settings, create_app
from geosim.catalog import Job as JobRow
from geosim.catalog import Project as ProjectRow
from geosim.catalog import SpatialFrameRow
from geosim.storage import BULK_STORES


@pytest.fixture
def settings(tmp_path):
    """No-service embedded tier: SQLite in-memory + temp storage root + inline runner."""
    return Settings(storage_root=tmp_path / "storage")


@pytest.fixture
def client(settings):
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


def test_m0_exit_create_project_then_job_streams_to_succeeded(client, settings):
    """The whole M0 walking skeleton in one flow (ROADMAP.md M0 Exit)."""
    session_factory = client.app.state.session_factory

    # 1) Create a project via the FastAPI app.
    resp = client.post("/projects", json={"name": "M0 Exit"})
    assert resp.status_code == 201
    proj = resp.json()
    pid = proj["id"]
    assert pid.startswith("prj_")

    # 2a) A catalog row exists (project + its 1:1 SpatialFrame row, doc 04 §2.4).
    with session_factory() as s:
        row = s.get(ProjectRow, pid)
        assert row is not None
        assert row.name == "M0 Exit"
        frame_row = s.execute(
            select(SpatialFrameRow).where(SpatialFrameRow.project_id == pid)
        ).scalar_one()
        assert frame_row.project_id == pid

    # 2b) A project directory exists on disk with the full bulk-store tree (doc 04 §3).
    proj_dir = settings.resolved_storage_root() / pid
    assert proj_dir.is_dir()
    for store in BULK_STORES:
        assert (proj_dir / store).is_dir(), f"missing bulk store {store!r}"
    assert (proj_dir / "frame.json").is_file()

    # 3) Enqueue the trivial demo job (doc 04 §9.4).
    enq = client.post(f"/projects/{pid}/jobs:demo", json={"steps": 4})
    assert enq.status_code == 202
    jid = enq.json()["job_id"]
    assert jid.startswith("job_")

    # 4) Progress streams over the WS endpoint to completion (doc 04 §9.2 WS).
    events: list[dict] = []
    with client.websocket_connect(f"/jobs/{jid}/progress") as ws:
        try:
            while True:
                events.append(ws.receive_json())
        except Exception:
            pass

    assert events, "expected progress events to stream over the WS"
    progresses = [e["progress"] for e in events]
    assert progresses == sorted(progresses), "progress must be monotonic 0→1"
    last = events[-1]
    assert last["status"] == "succeeded"
    assert last["progress"] == 1.0

    # The terminal state is also durable in the catalog (source of truth across reload).
    with session_factory() as s:
        job_row = s.get(JobRow, jid)
        assert job_row is not None
        assert job_row.project_id == pid
        assert job_row.status == "succeeded"
        assert job_row.progress == 1.0
        assert json.loads(job_row.result_json) == {"steps": 4, "done": True}
