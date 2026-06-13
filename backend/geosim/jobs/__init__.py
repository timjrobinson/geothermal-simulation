"""Async job runners (doc 04 §9.4 + the ``jobs`` table, doc 04 §2.4).

A single :class:`JobRunner` contract — ``enqueue(kind, params, fn) -> job_id`` plus a
job-state model matching the doc-04 ``jobs`` table — with two interchangeable executors:
:class:`InlineJobRunner` (synchronous no-service fallback) and :class:`RQJobRunner` (the
RQ + Redis async tier; Redis never required at import/test time). The job fn receives a
:class:`ProgressReporter` (``report``/``cancelled``) and pushes over a
:class:`ProgressChannel` a WS endpoint consumes.
"""

from .runner import (
    Cancelled,
    InlineJobRunner,
    JobFn,
    JobRunner,
    JobState,
    JobStatus,
    ProgressChannel,
    ProgressEvent,
    ProgressReporter,
    RQJobRunner,
    now_ms,
)

__all__ = [
    "JobStatus",
    "JobState",
    "ProgressEvent",
    "ProgressChannel",
    "ProgressReporter",
    "Cancelled",
    "JobFn",
    "JobRunner",
    "InlineJobRunner",
    "RQJobRunner",
    "now_ms",
]
