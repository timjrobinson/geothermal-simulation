"""Pydantic request/response shapes for the serving API (doc 04 §9.2, §9.3).

These mirror the doc-04 §9.2 wire shapes exactly: ``ProjectSummary``/``Project`` for the
project CRUD endpoints, the ``{name, frame?}`` create/patch bodies, and the job shape the
``GET /jobs/{jid}`` + WS ``{status, progress, message}`` (doc 04 §9.4) endpoints emit. The
``frame`` object is the doc-01 :class:`~geosim.spatial.frame.SpatialFrame` serialized by
:mod:`geosim.api.frame_io`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

__all__ = [
    "ProjectCreate",
    "ProjectPatch",
    "ProjectSummary",
    "Project",
    "JobOut",
    "DemoJobRequest",
    "EnqueueResponse",
]


class ProjectCreate(BaseModel):
    """``POST /projects`` body (doc 04 §9.2): ``{name, frame?}``."""

    name: str
    description: str | None = None
    frame: dict[str, Any] | None = None


class ProjectPatch(BaseModel):
    """``PATCH /projects/{pid}`` body (doc 04 §9.2): ``{name?, frame?}`` (frame = georeference)."""

    name: str | None = None
    description: str | None = None
    frame: dict[str, Any] | None = None


class ProjectSummary(BaseModel):
    """``GET /projects`` list element (doc 04 §9.2 ``[ProjectSummary]``)."""

    id: str
    name: str
    description: str | None = None
    created_at: int
    updated_at: int


class Project(ProjectSummary):
    """``GET /projects/{pid}`` (doc 04 §9.2): a project incl. its SpatialFrame."""

    storage_root: str
    frame: dict[str, Any]


class JobOut(BaseModel):
    """``GET /jobs/{jid}`` (doc 04 §9.2) — the durable ``jobs`` row shape (doc 04 §2.4)."""

    id: str
    kind: str
    project_id: str | None = None
    status: str
    progress: float
    message: str | None = None
    result: Any = None
    error: str | None = None
    created_at: int
    started_at: int | None = None
    finished_at: int | None = None


class DemoJobRequest(BaseModel):
    """``POST /projects/{pid}/jobs:demo`` body — a trivial 0→1 progress demo (doc 04 §9.4)."""

    steps: int = Field(default=5, ge=1, le=100)


class EnqueueResponse(BaseModel):
    """The ``{job_id}`` an async endpoint returns immediately (doc 04 §9.4 job pattern)."""

    job_id: str
