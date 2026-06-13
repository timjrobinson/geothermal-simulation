"""Serving API — the FastAPI app + DI'd foundation services (doc 04 §9, doc 08 §7.1).

This package is the HTTP/WS surface over the four foundation packages: it wires
:mod:`geosim.catalog` (metadata index), :mod:`geosim.storage` (bulk-store directories),
:mod:`geosim.jobs` (async runner), and :mod:`geosim.plugins` (capabilities) behind one
``create_app(settings)`` factory. Defaults select the no-service embedded tier — SQLite
in-memory catalog, a temp ``storage_root``, and an :class:`~geosim.jobs.InlineJobRunner` —
so the app runs with no Docker/Redis/Postgres (doc 04 §2.1, §3, §9.4).

Public surface: :func:`create_app`, :class:`Settings`, and the doc-04 §9.2/§9.3 wire shapes.
"""

from __future__ import annotations

from .app import Settings, create_app
from .schemas import (
    DemoJobRequest,
    EnqueueResponse,
    JobOut,
    Project,
    ProjectCreate,
    ProjectPatch,
    ProjectSummary,
)

__all__ = [
    "create_app",
    "Settings",
    "ProjectCreate",
    "ProjectPatch",
    "ProjectSummary",
    "Project",
    "JobOut",
    "DemoJobRequest",
    "EnqueueResponse",
]
