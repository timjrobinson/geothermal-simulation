"""Async job model and runners (doc 04 §9.4 + the ``jobs`` table, doc 04 §2.4).

One stable **JobRunner contract** across executors so swapping the executor is *not*
an API change (doc 04 §9.4): ``enqueue(kind, params, fn) -> job_id``. The job-state
model — ``queued|running|succeeded|failed|cancelled``, ``progress`` 0..1, ``message``,
``result``, ``error`` and the three timestamps — mirrors the doc-04 §2.4 ``jobs`` table
columns exactly (the durable source of truth; a job survives a page reload because the
client refetches ``GET /jobs/{jid}`` or reconnects to ``/jobs/{jid}/progress``).

Two executors implement the same contract (doc 04 §9.4 table):

- :class:`InlineJobRunner` — runs the fn synchronously in-process. This is the
  documented BackgroundTasks / no-service embedded fallback tier.
- :class:`RQJobRunner` — enqueues onto an RQ + Redis worker pool (the day-one async
  tier). Redis/rq are imported lazily and **never required** at import or test time.

The job fn receives a :class:`ProgressReporter` — ``report(progress, message)`` pushes
an update over the job's :class:`ProgressChannel` (the in-memory pub/sub a WS endpoint
consumes; the FastAPI WS endpoint itself is built by the API agent against this
contract) and ``cancelled()`` is the cooperative-cancellation check.
"""

from __future__ import annotations

import threading
import time
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from queue import Empty, Queue
from typing import Any

from ulid import ULID

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


def now_ms() -> int:
    """Epoch milliseconds — the timestamp unit on every catalog row (doc 04 §2.4)."""
    return int(time.time() * 1000)


class JobStatus(str, Enum):
    """The ``jobs.status`` enum (doc 04 §2.4)."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Terminal states — a job in one of these is finished and immutable.
_TERMINAL = frozenset({JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED})


class Cancelled(Exception):
    """Raised inside a job fn to cooperatively abort → job ends ``cancelled``.

    A fn may also simply ``return`` after observing ``reporter.cancelled()``; either
    way the runner records the terminal ``cancelled`` state.
    """


@dataclass
class ProgressEvent:
    """A single update pushed over a :class:`ProgressChannel` (the WS payload shape:
    ``{status, progress, message}`` per doc 04 §9.2, plus ``job_id``)."""

    job_id: str
    status: JobStatus
    progress: float
    message: str | None = None


@dataclass
class JobState:
    """In-memory mirror of one ``jobs`` row (doc 04 §2.4).

    The column set is intentionally 1:1 with the table so a persistence layer (the
    catalog agent) can serialize this directly: ``params``/``result``/``error`` map to
    the ``*_json`` columns; ``created_at``/``started_at``/``finished_at`` are epoch-ms.
    """

    id: str
    kind: str
    project_id: str | None = None
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    message: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    created_at: int = field(default_factory=now_ms)
    started_at: int | None = None
    finished_at: int | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL


class ProgressChannel:
    """Thread-safe in-memory pub/sub for one job's progress (doc 04 §9.4).

    The WS endpoint (built by the API agent) consumes this channel; in-process callers
    poll it directly. It serves two access patterns over the *same* event stream:

    - **callbacks** — ``subscribe(cb)`` registers a callable invoked on every publish;
    - **polling/iteration** — ``events()`` blocks for events until the channel closes,
      yielding an async-free iterable a WS handler can drive in a thread.

    The channel buffers all events so a late subscriber/poller still observes the full
    0→1 progression and the terminal event (the durable ``jobs`` row remains the source
    of truth for clients that reconnect after the channel is gone).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[ProgressEvent] = []
        self._callbacks: list[Callable[[ProgressEvent], None]] = []
        self._queues: list[Queue[ProgressEvent | None]] = []
        self._closed = False

    def publish(self, event: ProgressEvent) -> None:
        with self._lock:
            self._events.append(event)
            callbacks = list(self._callbacks)
            queues = list(self._queues)
        for cb in callbacks:
            cb(event)
        for q in queues:
            q.put(event)

    def subscribe(self, callback: Callable[[ProgressEvent], None]) -> None:
        """Register ``callback``; it is first replayed the buffered backlog so a late
        subscriber sees the full history, then fired on every subsequent publish."""
        with self._lock:
            backlog = list(self._events)
            self._callbacks.append(callback)
        for event in backlog:
            callback(event)

    @property
    def history(self) -> list[ProgressEvent]:
        """Snapshot of every event published so far (buffered)."""
        with self._lock:
            return list(self._events)

    def events(self, timeout: float | None = None) -> Iterator[ProgressEvent]:
        """Blocking iterator over events; replays the backlog then streams live ones
        until the channel is :meth:`close`\\ d. ``timeout`` (seconds) bounds the wait
        between events, after which iteration stops (so a WS handler never hangs)."""
        q: Queue[ProgressEvent | None] = Queue()
        with self._lock:
            backlog = list(self._events)
            if self._closed:
                yield from backlog
                return
            self._queues.append(q)
        yield from backlog
        while True:
            try:
                item = q.get(timeout=timeout)
            except Empty:
                return
            if item is None:  # close sentinel
                return
            yield item

    def close(self) -> None:
        """Signal end-of-stream to all pollers (called by the runner on terminal state)."""
        with self._lock:
            self._closed = True
            queues = list(self._queues)
        for q in queues:
            q.put(None)


class ProgressReporter:
    """Handed to a job fn: progress reporting + cooperative cancellation (doc 04 §9.4).

    ``report(progress, message)`` clamps ``progress`` to 0..1, updates the live
    :class:`JobState`, and publishes a :class:`ProgressEvent`. ``cancelled()`` returns
    True once cancellation was requested — the fn must check it at safe points and stop
    (raise :class:`Cancelled` or return) to abort cooperatively.
    """

    def __init__(self, state: JobState, channel: ProgressChannel,
                 cancel_event: threading.Event) -> None:
        self._state = state
        self._channel = channel
        self._cancel_event = cancel_event

    def report(self, progress: float, message: str | None = None) -> None:
        p = 0.0 if progress < 0.0 else 1.0 if progress > 1.0 else float(progress)
        self._state.progress = p
        if message is not None:
            self._state.message = message
        self._channel.publish(ProgressEvent(
            job_id=self._state.id,
            status=self._state.status,
            progress=p,
            message=message if message is not None else self._state.message,
        ))

    def cancelled(self) -> bool:
        """True once cancellation has been requested for this job."""
        return self._cancel_event.is_set()

    def raise_if_cancelled(self) -> None:
        """Convenience: raise :class:`Cancelled` if cancellation was requested."""
        if self._cancel_event.is_set():
            raise Cancelled


# A job fn takes the bound params dict and a ProgressReporter, returns a result.
JobFn = Callable[[dict[str, Any], ProgressReporter], Any]


class JobRunner:
    """The one stable executor contract (doc 04 §9.4).

    ``enqueue(kind, params, fn)`` creates a ``queued`` :class:`JobState`, schedules
    ``fn(params, reporter)`` on the executor and returns the kind-prefixed ULID job id
    immediately. Lifecycle/introspection methods (``get``, ``channel``, ``cancel``,
    ``wait``) are shared so a WS/REST layer is identical across executors.
    """

    _JOB_PREFIX = "job_"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobState] = {}
        self._channels: dict[str, ProgressChannel] = {}
        self._cancels: dict[str, threading.Event] = {}

    # ---------------------------------------------------------------- contract

    def enqueue(self, kind: str, params: dict[str, Any], fn: JobFn,
                *, project_id: str | None = None) -> str:
        """Register a job and dispatch it; return its job id immediately."""
        job_id = self._JOB_PREFIX + str(ULID())
        state = JobState(id=job_id, kind=kind, project_id=project_id,
                         params=dict(params or {}))
        channel = ProgressChannel()
        cancel_event = threading.Event()
        with self._lock:
            self._jobs[job_id] = state
            self._channels[job_id] = channel
            self._cancels[job_id] = cancel_event
        self._dispatch(state, channel, cancel_event, fn)
        return job_id

    def get(self, job_id: str) -> JobState | None:
        """The live :class:`JobState` mirror (the ``jobs`` row), or None."""
        with self._lock:
            return self._jobs.get(job_id)

    def channel(self, job_id: str) -> ProgressChannel | None:
        """The job's :class:`ProgressChannel` (for a WS/poll subscriber), or None."""
        with self._lock:
            return self._channels.get(job_id)

    def cancel(self, job_id: str) -> bool:
        """Request cooperative cancellation (doc 04 §9.2 ``POST /jobs/{jid}:cancel``).

        Sets the cancel flag the job fn observes via ``reporter.cancelled()``. A job
        that has not started running yet is marked ``cancelled`` immediately. Returns
        False for an unknown or already-terminal job.
        """
        with self._lock:
            state = self._jobs.get(job_id)
            event = self._cancels.get(job_id)
            channel = self._channels.get(job_id)
            if state is None or event is None or state.is_terminal:
                return False
            event.set()
            if state.status is JobStatus.QUEUED:
                self._finish(state, JobStatus.CANCELLED, channel)
            return True

    def wait(self, job_id: str, timeout: float | None = None) -> JobState | None:
        """Block until the job reaches a terminal state (default impl: inline = already
        terminal). Overridden by async executors. Returns the final state."""
        return self.get(job_id)

    # ---------------------------------------------------------------- internals

    def _dispatch(self, state: JobState, channel: ProgressChannel,
                  cancel_event: threading.Event, fn: JobFn) -> None:
        raise NotImplementedError

    def _run(self, state: JobState, channel: ProgressChannel,
             cancel_event: threading.Event, fn: JobFn) -> None:
        """Execute ``fn`` and drive the state machine (shared by all executors)."""
        if cancel_event.is_set():
            self._finish(state, JobStatus.CANCELLED, channel)
            return
        state.status = JobStatus.RUNNING
        state.started_at = now_ms()
        reporter = ProgressReporter(state, channel, cancel_event)
        channel.publish(ProgressEvent(state.id, JobStatus.RUNNING, state.progress,
                                      state.message))
        try:
            result = fn(state.params, reporter)
        except Cancelled:
            self._finish(state, JobStatus.CANCELLED, channel)
            return
        except Exception as exc:  # noqa: BLE001 — any fn failure → failed job
            state.error = "".join(traceback.format_exception(exc)).strip()
            self._finish(state, JobStatus.FAILED, channel, message=str(exc))
            return
        if cancel_event.is_set():
            self._finish(state, JobStatus.CANCELLED, channel)
            return
        state.result = result
        self._finish(state, JobStatus.SUCCEEDED, channel)

    def _finish(self, state: JobState, status: JobStatus, channel: ProgressChannel | None,
                *, message: str | None = None) -> None:
        state.status = status
        state.finished_at = now_ms()
        if status is JobStatus.SUCCEEDED:
            state.progress = 1.0
        if message is not None:
            state.message = message
        if channel is not None:
            channel.publish(ProgressEvent(state.id, status, state.progress, state.message))
            channel.close()


class InlineJobRunner(JobRunner):
    """Synchronous in-process executor (doc 04 §9.4 inline / BackgroundTasks fallback).

    ``enqueue`` runs the fn to completion before returning, so the returned job is
    already terminal — this is the no-service embedded tier paired with the SQLite
    fallback. The contract is identical to the RQ tier, so the WS/REST layer is unchanged.
    """

    def _dispatch(self, state: JobState, channel: ProgressChannel,
                  cancel_event: threading.Event, fn: JobFn) -> None:
        self._run(state, channel, cancel_event, fn)


class RQJobRunner(JobRunner):
    """RQ + Redis executor stub (doc 04 §9.4 — the day-one async tier).

    Enqueues ``fn`` onto an RQ queue backed by Redis at ``redis_url``. ``rq`` and
    ``redis`` are imported **lazily inside** ``enqueue`` so neither is required at import
    or test time (HARD RULE: no Redis at test time). With no ``redis_url`` configured
    this raises on ``enqueue`` rather than silently degrading — callers select
    :class:`InlineJobRunner` for the no-service path.

    NOTE: this is a stub. The worker process (a separate entrypoint, built later) is what
    actually calls :meth:`_run`; here we only register the durable :class:`JobState` and
    hand the callable to RQ. The catalog persistence (writing the ``jobs`` row the worker
    updates) is owned by the catalog/API agents against the same :class:`JobState` shape.
    """

    def __init__(self, redis_url: str | None = None, *, queue_name: str = "geosim") -> None:
        super().__init__()
        self.redis_url = redis_url
        self.queue_name = queue_name

    def _dispatch(self, state: JobState, channel: ProgressChannel,
                  cancel_event: threading.Event, fn: JobFn) -> None:
        if not self.redis_url:
            raise RuntimeError(
                "RQJobRunner requires a Redis URL; use InlineJobRunner for the "
                "no-service fallback (doc 04 §9.4)."
            )
        # Lazy imports — rq/redis are never touched at import or test time.
        from redis import Redis  # type: ignore[import-untyped]
        from rq import Queue  # type: ignore[import-untyped]

        connection = Redis.from_url(self.redis_url)
        queue = Queue(self.queue_name, connection=connection)
        # The worker re-materializes state/channel and calls self._run; we pass the
        # durable identity + params so the worker can rebuild the JobState.
        queue.enqueue(
            _rq_job_entrypoint,
            job_id=state.id,
            kwargs={
                "job_id": state.id,
                "kind": state.kind,
                "project_id": state.project_id,
                "params": state.params,
                "fn": fn,
            },
        )


def _rq_job_entrypoint(*, job_id: str, kind: str, project_id: str | None,
                       params: dict[str, Any], fn: JobFn) -> Any:
    """Module-level callable RQ invokes inside a worker (must be importable, not a
    closure). Rebuilds a :class:`JobState`/:class:`ProgressChannel` and runs the fn via
    a throwaway :class:`InlineJobRunner` so the exact same state machine applies."""
    runner = InlineJobRunner()
    state = JobState(id=job_id, kind=kind, project_id=project_id, params=dict(params))
    channel = ProgressChannel()
    cancel_event = threading.Event()
    runner._run(state, channel, cancel_event, fn)  # noqa: SLF001 — internal worker glue
    return state.result
