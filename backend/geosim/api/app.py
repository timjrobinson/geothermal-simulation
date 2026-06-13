"""FastAPI app factory + dependency-injected services (doc 04 §9, doc 08 §7.1).

``create_app(settings)`` wires the three pluggable foundation services behind FastAPI
dependencies so the whole stack runs with **no Docker/Redis/Postgres** (HARD RULE):

- **catalog** — a SQLAlchemy session factory (default: SQLite in-memory, doc 04 §2.1).
- **storage** — a ``storage_root`` whose per-project bulk-store tree is materialized via
  :func:`geosim.storage.ensure_project_layout` (doc 04 §3).
- **jobs** — a :class:`~geosim.jobs.JobRunner` (default: :class:`InlineJobRunner`, the
  no-service tier, doc 04 §9.4). The same contract swaps to RQ+Redis behind a flag.

Endpoints follow the doc-04 §9.2 shapes: project CRUD (each create materializes the
project directory + catalog rows + doc-01 ``SpatialFrame``), ``GET /api/capabilities``
(straight from :meth:`PluginRegistry.capabilities`, doc 08 §7.1), a trivial demo job
(``POST /projects/{pid}/jobs:demo`` → ``{job_id}``), ``GET /jobs/{jid}``, and the
``WS /jobs/{jid}/progress`` stream (doc 04 §9.4). CORS is opened for the Vite dev origin.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from geosim.catalog import (
    IdKind,
    Job,
    SpatialFrameRow,
    create_all,
    make_engine,
    new_id,
    session_factory,
)
from geosim.catalog import (
    Project as ProjectRow,
)
from geosim.jobs import InlineJobRunner, JobRunner, JobState, ProgressReporter
from geosim.plugins import get_registry
from geosim.storage import ensure_project_layout

from .frame_io import frame_from_dict, frame_from_row, frame_row_kwargs, frame_to_dict
from .property_models import build_property_model_router
from .schemas import (
    DemoJobRequest,
    EnqueueResponse,
    JobOut,
    Project,
    ProjectCreate,
    ProjectPatch,
    ProjectSummary,
)

__all__ = ["Settings", "create_app"]

# The Vite dev server origins (doc 06) — CORS-allowed so the SPA can call the API.
VITE_DEV_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


@dataclass
class Settings:
    """Injected configuration for :func:`create_app` (doc 04 §9).

    All defaults select the no-service embedded tier: a shared SQLite in-memory catalog
    (doc 04 §2.1), a temp ``storage_root`` (doc 04 §3), and an :class:`InlineJobRunner`
    (doc 04 §9.4) — so the app runs with no Docker/Redis/Postgres. The flags only widen to
    Postgres/RQ when explicitly provided.
    """

    database_url: str | None = None  # None ⇒ SQLite in-memory fallback (doc 04 §2.1)
    storage_root: str | Path | None = None  # None ⇒ a temp dir (doc 04 §3)
    job_runner: JobRunner | None = None  # None ⇒ InlineJobRunner (doc 04 §9.4)
    cors_origins: tuple[str, ...] = VITE_DEV_ORIGINS
    discover_plugins: bool = False  # entry-point discovery off by default (deterministic tests)
    _tempdir: Any = field(default=None, repr=False)

    def resolved_storage_root(self) -> Path:
        if self.storage_root is not None:
            root = Path(self.storage_root)
            root.mkdir(parents=True, exist_ok=True)
            return root
        # Hold the TemporaryDirectory on the settings so it lives for the app's lifetime.
        self._tempdir = tempfile.TemporaryDirectory(prefix="geosim-storage-")
        return Path(self._tempdir.name)


def _project_to_summary(row: ProjectRow) -> ProjectSummary:
    return ProjectSummary(
        id=row.id,
        name=row.name,
        description=row.description,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _project_to_full(row: ProjectRow, frame_dict: dict[str, Any]) -> Project:
    return Project(
        id=row.id,
        name=row.name,
        description=row.description,
        created_at=row.created_at,
        updated_at=row.updated_at,
        storage_root=row.storage_root,
        frame=frame_dict,
    )


def _job_state_to_out(state: JobState) -> JobOut:
    return JobOut(
        id=state.id,
        kind=state.kind,
        project_id=state.project_id,
        status=state.status.value,
        progress=state.progress,
        message=state.message,
        result=state.result,
        error=state.error,
        created_at=state.created_at,
        started_at=state.started_at,
        finished_at=state.finished_at,
    )


def _job_row_to_out(row: Job) -> JobOut:
    return JobOut(
        id=row.id,
        kind=row.kind,
        project_id=row.project_id,
        status=row.status,
        progress=row.progress,
        message=row.message,
        result=json.loads(row.result_json) if row.result_json else None,
        error=row.error_json,
        created_at=row.created_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
    )


def _demo_job_fn(params: dict[str, Any], reporter: ProgressReporter) -> dict[str, Any]:
    """A trivial job that walks progress 0→1 over ``steps`` (doc 04 §9.4 demo).

    Pushes a :class:`~geosim.jobs.ProgressEvent` per step over the job's
    :class:`ProgressChannel` (the WS endpoint consumes it) and checks cooperative
    cancellation. The InlineJobRunner runs this synchronously so the returned job is
    already terminal (no service required).
    """
    steps = int(params.get("steps", 5))
    for i in range(1, steps + 1):
        reporter.raise_if_cancelled()
        reporter.report(i / steps, f"step {i}/{steps}")
    return {"steps": steps, "done": True}


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app with DI'd catalog/storage/jobs (doc 04 §9).

    With ``settings=None`` everything defaults to the no-service tier. The catalog engine,
    storage root, and job runner are stored on ``app.state`` and exposed through FastAPI
    dependencies so tests (and the prod build) can override them.
    """
    settings = settings or Settings()

    engine = make_engine(settings.database_url)
    create_all(engine)  # tests use create_all; prod uses Alembic (doc 04 §2.1)
    Session_ = session_factory(engine)
    storage_root = settings.resolved_storage_root()
    runner: JobRunner = settings.job_runner or InlineJobRunner()
    registry = get_registry()
    if settings.discover_plugins:
        registry.discover_entry_points()

    app = FastAPI(title="GeoSim API", version="0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Stash services on app.state so dependencies + WS handlers can reach them.
    app.state.engine = engine
    app.state.session_factory = Session_
    app.state.storage_root = storage_root
    app.state.job_runner = runner
    app.state.settings = settings

    def get_session() -> Any:
        session = Session_()
        try:
            yield session
        finally:
            session.close()

    def get_runner() -> JobRunner:
        return runner

    # FastAPI DI markers. These must be evaluated at runtime (not via the string
    # annotations ``from __future__ import annotations`` produces), so they live in the
    # parameter *default*, which means the B008 "call in default" lint must be suppressed
    # on each route (the idiomatic FastAPI pattern with a function-local dependency).
    session_dep = Depends(get_session)
    runner_dep = Depends(get_runner)

    # PropertyModel read surface (volume/slice/zarr) — shares the catalog session +
    # storage_root DI off app.state (doc 04 §9.2/§9.3, doc 06 §1.3/§12).
    app.include_router(build_property_model_router(session_dep))

    # ──────────────────────────────── capabilities (doc 08 §7.1) ────────────────────────────
    @app.get("/api/capabilities")
    def capabilities() -> dict[str, Any]:
        """The single backend→frontend contract (doc 08 §7.1).

        Comes straight from :meth:`PluginRegistry.capabilities`; property types flow from
        the doc-01 §5 registry, so this is populated even with zero plugins loaded.
        """
        return registry.capabilities()

    # ──────────────────────────────── projects (doc 04 §9.2) ────────────────────────────────
    @app.get("/projects", response_model=list[ProjectSummary])
    def list_projects(session: Session = session_dep) -> list[ProjectSummary]:
        rows = session.query(ProjectRow).order_by(ProjectRow.created_at).all()
        return [_project_to_summary(r) for r in rows]

    @app.post("/projects", response_model=Project, status_code=201)
    def create_project(
        body: ProjectCreate, session: Session = session_dep
    ) -> Project:
        """Create a project: catalog rows + bulk-store directory + doc-01 SpatialFrame.

        The project row, its 1:1 ``SpatialFrameRow`` (doc 04 §2.4), and the on-disk
        ``<storage_root>/<pid>/`` bulk-store tree (doc 04 §3) are all created in one go.
        """
        pid = new_id(IdKind.PROJECT)
        frame = frame_from_dict(body.frame)

        # Materialize the bulk-store directory tree (doc 04 §3).
        layout = ensure_project_layout(storage_root, pid)

        row = ProjectRow(
            id=pid,
            name=body.name,
            description=body.description,
            storage_root=str(storage_root),
        )
        row.spatial_frame = SpatialFrameRow(
            project_id=pid, **frame_row_kwargs(frame)
        )
        session.add(row)
        session.commit()
        session.refresh(row)

        # Cache the frame next to the bulk stores (DB is canonical; doc 04 §3).
        layout.frame_json.write_text(json.dumps(frame_to_dict(frame), indent=2))
        return _project_to_full(row, frame_to_dict(frame))

    @app.get("/projects/{pid}", response_model=Project)
    def get_project(pid: str, session: Session = session_dep) -> Project:
        row = session.get(ProjectRow, pid)
        if row is None:
            raise HTTPException(status_code=404, detail="project not found")
        frame = frame_from_row(row.spatial_frame)
        return _project_to_full(row, frame_to_dict(frame))

    @app.patch("/projects/{pid}", response_model=Project)
    def patch_project(
        pid: str, body: ProjectPatch, session: Session = session_dep
    ) -> Project:
        """Edit a project (doc 04 §9.2). A ``frame`` edit is a doc-01 georeference promote;
        bulk arrays are never reprocessed — only frame metadata changes (doc 01 §2)."""
        row = session.get(ProjectRow, pid)
        if row is None:
            raise HTTPException(status_code=404, detail="project not found")
        if body.name is not None:
            row.name = body.name
        if body.description is not None:
            row.description = body.description
        if body.frame is not None:
            frame = frame_from_dict(body.frame)
            for key, value in frame_row_kwargs(frame).items():
                setattr(row.spatial_frame, key, value)
        session.commit()
        session.refresh(row)
        frame = frame_from_row(row.spatial_frame)
        return _project_to_full(row, frame_to_dict(frame))

    @app.delete("/projects/{pid}", status_code=204)
    def delete_project(pid: str, session: Session = session_dep) -> None:
        row = session.get(ProjectRow, pid)
        if row is None:
            raise HTTPException(status_code=404, detail="project not found")
        session.delete(row)  # ON DELETE CASCADE clears children (doc 04 §2.4)
        session.commit()

    # ──────────────────────────────── jobs (doc 04 §9.2, §9.4) ───────────────────────────────
    @app.post(
        "/projects/{pid}/jobs:demo", response_model=EnqueueResponse, status_code=202
    )
    def enqueue_demo_job(
        pid: str,
        session: Session = session_dep,
        job_runner: JobRunner = runner_dep,
        body: DemoJobRequest | None = None,
    ) -> EnqueueResponse:
        """Enqueue a trivial demo job → ``{job_id}`` (doc 04 §9.4 job pattern).

        Writes the durable ``jobs`` row (the source of truth, doc 04 §2.4), enqueues onto
        the runner, then persists the terminal state back to the row. With the inline
        runner the job is already terminal on return, reaching ``progress=1.0`` /
        ``status=succeeded``.
        """
        if session.get(ProjectRow, pid) is None:
            raise HTTPException(status_code=404, detail="project not found")
        params = (body or DemoJobRequest()).model_dump()
        job_id = job_runner.enqueue("demo", params, _demo_job_fn, project_id=pid)
        _persist_job(session, job_runner, job_id, pid, params)
        return EnqueueResponse(job_id=job_id)

    @app.get("/jobs/{jid}", response_model=JobOut)
    def get_job(
        jid: str,
        session: Session = session_dep,
        job_runner: JobRunner = runner_dep,
    ) -> JobOut:
        """Fetch a job (doc 04 §9.2). Prefer the live runner state; fall back to the durable
        ``jobs`` row so a job survives an API restart / page reload (doc 04 §9.4)."""
        state = job_runner.get(jid)
        if state is not None:
            return _job_state_to_out(state)
        row = session.get(Job, jid)
        if row is None:
            raise HTTPException(status_code=404, detail="job not found")
        return _job_row_to_out(row)

    @app.post("/jobs/{jid}:cancel", status_code=202)
    def cancel_job(
        jid: str, job_runner: JobRunner = runner_dep
    ) -> dict[str, bool]:
        """Request cooperative cancellation (doc 04 §9.2 ``POST /jobs/{jid}:cancel``)."""
        return {"cancelled": job_runner.cancel(jid)}

    @app.websocket("/jobs/{jid}/progress")
    async def job_progress(websocket: WebSocket, jid: str) -> None:
        """Stream ``{status, progress, message}`` for a job (doc 04 §9.2 WS).

        Subscribes to the job's :class:`ProgressChannel` (doc 04 §9.4) and replays the
        buffered backlog, so even an inline job that already finished still delivers its
        full 0→1 progression and terminal event to a late-connecting client.
        """
        await websocket.accept()
        runner_: JobRunner = websocket.app.state.job_runner
        channel = runner_.channel(jid)
        if channel is None:
            await websocket.close(code=1008)
            return
        try:
            for event in channel.events(timeout=5.0):
                await websocket.send_json(
                    {
                        "status": event.status.value,
                        "progress": event.progress,
                        "message": event.message,
                    }
                )
            await websocket.close()
        except WebSocketDisconnect:  # pragma: no cover - client hangup
            pass

    return app


def _persist_job(
    session: Session,
    runner: JobRunner,
    job_id: str,
    project_id: str,
    params: dict[str, Any],
) -> None:
    """Write/refresh the durable ``jobs`` row from the runner state (doc 04 §2.4, §9.4).

    The ``jobs`` row is the durable source of truth a client refetches after a reload.
    With the inline runner the state is already terminal here, so the row captures the
    final ``status``/``progress``/``result``.
    """
    state = runner.get(job_id)
    row = session.get(Job, job_id)
    if row is None:
        row = Job(id=job_id, project_id=project_id, kind="demo", params_json=json.dumps(params))
        session.add(row)
    if state is not None:
        row.status = state.status.value
        row.progress = state.progress
        row.message = state.message
        row.result_json = json.dumps(state.result) if state.result is not None else None
        row.error_json = state.error
        row.started_at = state.started_at
        row.finished_at = state.finished_at
    else:  # pragma: no cover - async runner with no immediate state
        row.status = "queued"
    session.commit()
