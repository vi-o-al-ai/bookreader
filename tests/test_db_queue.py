"""Tests for bookreader.jobs: the SQLite JobStore and the worker queue implementations."""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest

from bookreader.jobs.db import JobStore, new_job_id
from bookreader.jobs.queue import InlineQueue, InProcessQueue, build_queue
from bookreader.settings import Settings
from bookreader.types import STAGE_ORDER, Job, JobError, JobOptions, JobStatus, Stage, StageState, UsageEvent


@pytest.fixture
def store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path / "data" / "bookreader.db")


def make_job(job_id: str | None = None, title: str = "Book", **kw: object) -> Job:
    return Job(id=job_id or new_job_id(), title=title, filename="book.txt", **kw)  # type: ignore[arg-type]


def test_schema_creation_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "data" / "bookreader.db"
    JobStore(path)
    JobStore(path)                                        # a second open must not fail or reset anything
    with sqlite3.connect(path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert {"jobs", "job_stages", "job_events", "usage_events"} <= tables
    assert mode == "wal"


def test_create_get_list_update(store: JobStore) -> None:
    first = store.create_job(make_job(title="First", options=JobOptions(music=False, cast_overrides={"Tobias": "v1"})), "{}")
    assert first.status == JobStatus.queued and first.created_at and first.updated_at
    assert first.options.music is False and first.options.cast_overrides == {"Tobias": "v1"}
    time.sleep(0.002)
    second = store.create_job(make_job(title="Second"), Settings.from_env({}).snapshot())
    assert [job.id for job in store.list_jobs()] == [second.id, first.id]
    assert [job.id for job in store.list_jobs(limit=1)] == [second.id]
    assert store.get_job("missing") is None
    assert '"data_dir"' in (store.settings_snapshot(second.id) or "")

    store.update_job(first.id, title="Renamed", source_sha256="abc", stage=Stage.render, error=JobError(stage="render", error_type="internal", message="boom", retryable=False))
    job = store.get_job(first.id)
    assert job is not None
    assert (job.title, job.source_sha256, job.stage) == ("Renamed", "abc", Stage.render)
    assert job.error is not None and job.error.message == "boom"
    assert job.updated_at >= first.updated_at
    with pytest.raises(ValueError, match="unknown job field"):
        store.update_job(first.id, bogus=1)

    store.set_status(first.id, JobStatus.running, stage="ingest")
    job = store.get_job(first.id)
    assert job is not None and job.status == JobStatus.running and job.stage == Stage.ingest and job.error is None and job.finished_at is None
    store.set_status(first.id, "done")
    job = store.get_job(first.id)
    assert job is not None and job.status == JobStatus.done and job.finished_at is not None


def test_stage_records_in_order(store: JobStore) -> None:
    job = store.create_job(make_job(), "{}")
    records = store.stage_records(job.id)
    assert [r.stage for r in records] == list(STAGE_ORDER)
    assert all(r.state == StageState.pending and r.attempts == 0 for r in records)

    store.set_stage(job.id, Stage.analyze, state=StageState.running, done=0, total=12, message="chunk 0/12", bump_attempts=True)
    store.set_stage(job.id, Stage.analyze, done=5, message="chunk 5/12")
    record = store.stage_records(job.id)[1]
    assert (record.state, record.done, record.total, record.message, record.attempts) == (StageState.running, 5, 12, "chunk 5/12", 1)
    assert record.started_at is not None and record.finished_at is None
    store.set_stage(job.id, "analyze", state="done")
    record = store.stage_records(job.id)[1]
    assert record.state == StageState.done and record.finished_at is not None

    for stage in (Stage.cast, Stage.render, Stage.finalize):
        store.set_stage(job.id, stage, state=StageState.done, done=3, total=3, bump_attempts=True)
    store.reset_stages_from(job.id, Stage.render)
    states = {r.stage: r for r in store.stage_records(job.id)}
    assert states[Stage.cast].state == StageState.done
    assert states[Stage.render].state == StageState.pending and states[Stage.render].done == 0 and states[Stage.render].attempts == 1
    assert states[Stage.finalize].state == StageState.pending and states[Stage.finalize].started_at is None


def test_events_pagination(store: JobStore) -> None:
    job = store.create_job(make_job(), "{}")
    ids = [store.append_event(job.id, "info", "ingest" if i % 2 else None, f"event {i}") for i in range(10)]
    assert ids == sorted(ids) and len(set(ids)) == 10
    page = store.events_after(job.id, 0, limit=4)
    assert [e.message for e in page] == ["event 0", "event 1", "event 2", "event 3"]
    assert page[1].stage == "ingest" and page[0].stage is None and page[0].level == "info" and page[0].job_id == job.id
    rest = store.events_after(job.id, page[-1].id, limit=100)
    assert [e.message for e in rest] == [f"event {i}" for i in range(4, 10)]
    assert store.events_after(job.id, rest[-1].id) == []


def test_usage_summary(store: JobStore) -> None:
    job = store.create_job(make_job(), "{}")
    store.add_usage(job.id, UsageEvent(capability="analysis", family="anthropic", unit_type="input_tokens", units=1000, cost_usd=0.005))
    store.add_usage(job.id, UsageEvent(capability="analysis", family="anthropic", unit_type="output_tokens", units=200, cost_usd=0.005, meta={"chunk": 0}))
    store.add_usage(job.id, UsageEvent(capability="tts", family="elevenlabs", unit_type="characters", units=300, cache_hit=True))
    summary = store.usage_summary(job.id)
    assert summary.calls == 2 and summary.cache_hits == 1
    assert summary.cost_usd == pytest.approx(0.01)
    assert summary.by_capability == {"analysis": {"input_tokens": 1000.0, "output_tokens": 200.0}, "tts": {"characters": 300.0}}
    assert store.usage_count(job.id) == 3
    assert store.usage_summary("missing").calls == 0


def test_requeue_orphans_and_cancel_flag(store: JobStore) -> None:
    running = store.create_job(make_job(), "{}")
    queued = store.create_job(make_job(), "{}")
    store.set_status(running.id, JobStatus.running, stage=Stage.render)
    store.set_stage(running.id, Stage.render, state=StageState.running)
    assert store.requeue_orphans() == [running.id]
    assert store.requeue_orphans() == []
    job = store.get_job(running.id)
    assert job is not None and job.status == JobStatus.queued
    assert {r.stage: r.state for r in store.stage_records(running.id)}[Stage.render] == StageState.pending
    assert store.get_job(queued.id).status == JobStatus.queued  # type: ignore[union-attr]

    assert store.cancel_requested(queued.id) is False
    store.request_cancel(queued.id)
    assert store.cancel_requested(queued.id) is True
    assert store.get_job(queued.id).cancel_requested is True  # type: ignore[union-attr]
    store.set_status(queued.id, JobStatus.cancelled, error=JobError(stage="render", error_type="cancelled", message="x", retryable=True))
    store.requeue(queued.id, from_stage=Stage.cast)
    job = store.get_job(queued.id)
    assert job is not None and job.status == JobStatus.queued and job.cancel_requested is False and job.error is None and job.finished_at is None
    assert store.cancel_requested("missing") is False


def test_delete_cascades(store: JobStore) -> None:
    job = store.create_job(make_job(), "{}")
    store.append_event(job.id, "info", None, "hello")
    store.add_usage(job.id, UsageEvent(capability="tts", family="mock", unit_type="characters", units=10))
    assert store.delete_job(job.id) is True
    assert store.delete_job(job.id) is False
    assert store.get_job(job.id) is None
    assert store.stage_records(job.id) == []
    assert store.events_after(job.id) == []
    assert store.usage_count(job.id) == 0


def test_thread_safety(store: JobStore) -> None:
    job = store.create_job(make_job(), "{}")
    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            for i in range(50):
                store.append_event(job.id, "info", "render", f"t{n} e{i}")
                store.set_stage(job.id, Stage.render, done=i, total=50)
        except BaseException as exc:  # noqa: BLE001 - surfaced through the assertion below
            errors.append(exc)
        finally:
            store.close()

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    assert not errors
    events = store.events_after(job.id, 0, limit=1000)
    assert len(events) == 400
    assert len({e.id for e in events}) == 400


def test_in_process_queue_runs_and_stop_interrupts() -> None:
    seen: list[str] = []
    interrupted = threading.Event()

    def run(job_id: str, stop: threading.Event) -> None:
        seen.append(job_id)
        if stop.wait(10):
            interrupted.set()

    q = InProcessQueue(run, workers=1)
    assert q.backend == "thread" and q.depth() == 0
    q.start()
    q.start()                                             # idempotent
    q.submit("job-1")
    deadline = time.monotonic() + 5
    while not seen and time.monotonic() < deadline:
        time.sleep(0.01)
    assert seen == ["job-1"]
    assert q.running() == ["job-1"]
    started = time.monotonic()
    q.stop(timeout=5)
    assert time.monotonic() - started < 5
    assert interrupted.is_set()
    assert q.running() == []


def test_in_process_queue_survives_crashing_job() -> None:
    seen: list[str] = []

    def run(job_id: str, stop: threading.Event) -> None:
        seen.append(job_id)
        if job_id == "bad":
            raise RuntimeError("boom")

    q = InProcessQueue(run, workers=2)
    q.start()
    q.submit("bad")
    q.submit("good")
    deadline = time.monotonic() + 5
    while len(seen) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    q.stop(timeout=5)
    assert sorted(seen) == ["bad", "good"]


def test_in_process_queue_dedupes_pending_ids_and_discard_skips() -> None:
    """A cancel-then-retry of a queued job submits the same id twice; it must run once. A deleted
    (discarded) id must be skipped by the worker rather than crash the loop."""
    gate = threading.Event()
    seen: list[str] = []

    def run(job_id: str, stop: threading.Event) -> None:
        seen.append(job_id)
        gate.wait(5)

    q = InProcessQueue(run, workers=2)
    q.submit("dup")
    q.submit("dup")                                       # duplicate while still waiting: no-op
    q.submit("gone")
    q.submit("other")
    assert q.depth() == 3
    assert q.discard("gone") is True and q.discard("gone") is False and q.discard("never") is False
    assert q.depth() == 2
    q.start()
    deadline = time.monotonic() + 5
    while len(seen) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.2)                                       # give a wrong implementation time to run "dup" again
    assert sorted(seen) == ["dup", "other"] and sorted(q.running()) == ["dup", "other"] and q.depth() == 0
    q.submit("dup")                                       # a running id may be queued again (retry after cancel)
    assert q.depth() == 1
    gate.set()
    deadline = time.monotonic() + 5
    while len(seen) < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    q.stop(timeout=5)
    assert seen.count("dup") == 2 and "gone" not in seen


def test_claim_moves_only_queued_jobs_to_running(store: JobStore) -> None:
    job = store.create_job(make_job(), "{}")
    assert store.claim(job.id) is True
    assert store.get_job(job.id).status == JobStatus.running  # type: ignore[union-attr]
    assert store.claim(job.id) is False                   # already running (another worker holds it)
    store.set_status(job.id, JobStatus.cancelled, error=JobError(stage="queued", error_type="cancelled", message="x", retryable=True))
    assert store.claim(job.id) is False                   # a stale queue entry of a cancelled job
    assert store.get_job(job.id).status == JobStatus.cancelled  # type: ignore[union-attr]
    assert store.claim(job.id, (JobStatus.queued, JobStatus.cancelled)) is True
    claimed = store.get_job(job.id)
    assert claimed is not None and claimed.status == JobStatus.running and claimed.error is None and claimed.finished_at is None
    assert store.claim("missing") is False


def test_inline_queue_is_synchronous_and_build_queue_switches() -> None:
    seen: list[str] = []
    q = InlineQueue(lambda job_id, stop: seen.append(job_id))
    q.start()
    q.submit("a")
    assert seen == ["a"] and q.depth() == 0 and q.backend == "inline"
    assert q.discard("a") is False
    q.stop()
    assert q.stop_event.is_set()

    base = Settings.from_env({})
    assert isinstance(build_queue(base.with_overrides(worker_mode="inline"), lambda j, s: None), InlineQueue)
    threaded = build_queue(base.with_overrides(worker_mode="thread", workers=3), lambda j, s: None)
    assert isinstance(threaded, InProcessQueue) and threaded.workers == 3
