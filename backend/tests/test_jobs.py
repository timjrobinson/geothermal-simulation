"""Tests for the jobs subpackage (doc 04 §9.4 + §2.4 jobs table).

No Redis/RQ used: every test exercises :class:`InlineJobRunner` or the channel/state
primitives. The RQ path is only checked for lazy-import / no-config behaviour.
"""

from __future__ import annotations

import threading

import pytest

from geosim.jobs import (
    Cancelled,
    InlineJobRunner,
    JobState,
    JobStatus,
    ProgressChannel,
    ProgressEvent,
    RQJobRunner,
)


def test_inline_success_reports_progress_and_captures_result():
    runner = InlineJobRunner()
    seen: list[float] = []

    def fn(params, reporter):
        reporter.report(0.0, "starting")
        for i in (1, 2, 3, 4):
            reporter.report(i / 4.0, f"step {i}")
        return {"answer": params["x"] * 2}

    job_id = runner.enqueue("transform", {"x": 21}, fn)
    state = runner.get(job_id)

    assert job_id.startswith("job_")
    assert state is not None
    assert state.status is JobStatus.SUCCEEDED
    assert state.result == {"answer": 42}
    assert state.progress == 1.0
    assert state.error is None
    assert state.kind == "transform"
    # timestamps populated (epoch-ms, doc 04 §2.4)
    assert state.created_at > 0
    assert state.started_at is not None and state.started_at >= state.created_at
    assert state.finished_at is not None and state.finished_at >= state.started_at

    # progress 0 -> 1 observed via the channel
    channel = runner.channel(job_id)
    assert channel is not None
    progresses = [e.progress for e in channel.history]
    assert progresses[0] == 0.0
    assert max(progresses) == 1.0
    assert any(e.status is JobStatus.RUNNING for e in channel.history)
    assert channel.history[-1].status is JobStatus.SUCCEEDED


def test_inline_failure_sets_failed_with_error():
    runner = InlineJobRunner()

    def fn(params, reporter):
        reporter.report(0.5, "halfway")
        raise ValueError("boom")

    job_id = runner.enqueue("ingest", {}, fn)
    state = runner.get(job_id)

    assert state is not None
    assert state.status is JobStatus.FAILED
    assert state.result is None
    assert state.error is not None
    assert "boom" in state.error
    assert "ValueError" in state.error
    assert state.finished_at is not None

    channel = runner.channel(job_id)
    assert channel is not None
    assert channel.history[-1].status is JobStatus.FAILED


def test_progress_events_observed_via_subscribe_callback():
    runner = InlineJobRunner()
    events: list[ProgressEvent] = []

    # subscribe BEFORE enqueue is not possible (no job yet); subscribe after, the
    # channel replays the full buffered backlog so nothing is missed.
    def fn(params, reporter):
        reporter.report(0.25, "a")
        reporter.report(0.75, "b")
        return "ok"

    job_id = runner.enqueue("fuse", {}, fn)
    channel = runner.channel(job_id)
    assert channel is not None
    channel.subscribe(events.append)

    statuses = [e.status for e in events]
    assert JobStatus.SUCCEEDED in statuses
    assert pytest.approx(0.25) in [e.progress for e in events]
    assert pytest.approx(0.75) in [e.progress for e in events]


def test_progress_clamped_to_unit_interval():
    runner = InlineJobRunner()

    def fn(params, reporter):
        reporter.report(-0.5, "under")
        reporter.report(2.0, "over")
        return None

    job_id = runner.enqueue("pyramid", {}, fn)
    channel = runner.channel(job_id)
    assert channel is not None
    vals = [e.progress for e in channel.history]
    assert min(vals) >= 0.0
    assert max(vals) <= 1.0


def test_cancellation_flips_flag_and_fn_returns_cancelled():
    runner = InlineJobRunner()
    observed_cancel: list[bool] = []

    def fn(params, reporter):
        # The runner injects cancellation via the cancel event; simulate a fn that is
        # told to cancel mid-flight by setting it through the event we capture below.
        params["cancel_event"].set()
        observed_cancel.append(reporter.cancelled())
        reporter.raise_if_cancelled()
        return "should not reach"

    cancel_event = threading.Event()
    # We can't reach into the runner's private event here, so verify the reporter-level
    # contract directly with a JobState + channel (the runner wires the same objects).
    state = JobState(id="job_x", kind="export")
    channel = ProgressChannel()
    from geosim.jobs import ProgressReporter

    reporter = ProgressReporter(state, channel, cancel_event)
    assert reporter.cancelled() is False
    cancel_event.set()
    assert reporter.cancelled() is True
    with pytest.raises(Cancelled):
        reporter.raise_if_cancelled()


def test_runner_cancel_on_cooperative_fn():
    runner = InlineJobRunner()
    started = threading.Event()
    release = threading.Event()
    result_box: dict[str, object] = {}

    def fn(params, reporter):
        started.set()
        release.wait(timeout=2.0)  # wait until the test requests cancel
        if reporter.cancelled():
            raise Cancelled
        return "done"

    # Run the job on a background thread so we can request cancel while it runs.
    def drive():
        result_box["job_id"] = runner.enqueue("gc", {}, fn)

    t = threading.Thread(target=drive)
    t.start()
    assert started.wait(timeout=2.0)
    # Job is RUNNING inline on thread t; request cancellation, then let fn proceed.
    job_id_holder: dict[str, str] = {}

    # Find the job id (enqueue hasn't returned yet on the inline runner since it blocks);
    # locate the single in-flight job.
    while not runner._jobs:  # noqa: SLF001 — test introspection
        pass
    [jid] = list(runner._jobs)  # noqa: SLF001
    job_id_holder["jid"] = jid
    assert runner.cancel(jid) is True
    release.set()
    t.join(timeout=2.0)

    state = runner.get(job_id_holder["jid"])
    assert state is not None
    assert state.status is JobStatus.CANCELLED
    assert state.result is None


def test_cancel_queued_unknown_and_terminal():
    runner = InlineJobRunner()
    # unknown job
    assert runner.cancel("job_nope") is False
    # already-terminal job cannot be cancelled
    jid = runner.enqueue("transform", {}, lambda p, r: 1)
    assert runner.get(jid).status is JobStatus.SUCCEEDED
    assert runner.cancel(jid) is False


def test_channel_events_iterator_after_close():
    channel = ProgressChannel()
    channel.publish(ProgressEvent("job_1", JobStatus.RUNNING, 0.5, "mid"))
    channel.publish(ProgressEvent("job_1", JobStatus.SUCCEEDED, 1.0, "done"))
    channel.close()
    collected = list(channel.events(timeout=0.5))
    assert [e.progress for e in collected] == [0.5, 1.0]
    assert collected[-1].status is JobStatus.SUCCEEDED


def test_channel_events_live_stream():
    channel = ProgressChannel()
    received: list[ProgressEvent] = []

    def consume():
        for e in channel.events(timeout=2.0):
            received.append(e)

    t = threading.Thread(target=consume)
    t.start()
    channel.publish(ProgressEvent("job_1", JobStatus.RUNNING, 0.3, "x"))
    channel.publish(ProgressEvent("job_1", JobStatus.SUCCEEDED, 1.0, "y"))
    channel.close()
    t.join(timeout=2.0)
    assert [e.progress for e in received] == [0.3, 1.0]


def test_jobstate_terminal_property():
    assert JobState(id="j", kind="k", status=JobStatus.QUEUED).is_terminal is False
    assert JobState(id="j", kind="k", status=JobStatus.RUNNING).is_terminal is False
    for s in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED):
        assert JobState(id="j", kind="k", status=s).is_terminal is True


def test_rq_runner_no_redis_url_raises_on_enqueue_not_import():
    # Importing/constructing must not require redis or rq.
    runner = RQJobRunner(redis_url=None)
    with pytest.raises(RuntimeError, match="Redis"):
        runner.enqueue("ingest", {}, lambda p, r: None)
