"""bookreader.pipeline.run - ``run_job``, the loop that drives one job through the stages.

``run_job`` needs only the SQLite store and the data directory, so any queue consumer (the
in-process worker, the CLI, an external queue) can call it. It merges the job's stored settings
snapshot into the current process settings, builds providers when none are injected, runs every
stage that is not already done (or from ``from_stage`` onward), and turns exceptions into a typed
``JobError`` with secrets scrubbed from the message.
"""
from __future__ import annotations

import logging
import re
import shutil
import threading
import traceback
from pathlib import Path

from bookreader.jobs.db import JobStore, new_job_id, now_iso
from bookreader.jobs.paths import JobPaths
from bookreader.pipeline.cache import ClipCache
from bookreader.pipeline.context import JobContext, attach_job_log, detach_job_log, job_logger
from bookreader.pipeline.stages import STAGE_FUNCTIONS
from bookreader.providers.base import Providers, build_providers
from bookreader.settings import Settings
from bookreader.types import (
    STAGE_ORDER,
    InputError,
    Job,
    JobCancelled,
    JobError,
    JobOptions,
    JobStatus,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
    Stage,
    StageState,
)
from bookreader.usage import UsageLedger, UsageRouter

log = logging.getLogger(__name__)

_API_KEY_RE = re.compile(r"sk-[A-Za-z0-9_-]{8,}")
MAX_STAGE_MESSAGE = 200


def scrub(message: str, secrets: dict[str, str]) -> str:
    """Remove API-key-looking tokens and every configured secret value from *message*."""
    text = _API_KEY_RE.sub("sk-***", message)
    for value in secrets.values():
        if value:
            text = text.replace(value, "***")
    return text


def classify(exc: BaseException) -> tuple[str, bool]:
    """``(error_type, retryable)`` for an exception escaping a stage."""
    if isinstance(exc, InputError):
        return "input", False
    if isinstance(exc, ProviderConfigError):
        return "config", False
    if isinstance(exc, ProviderTransientError):
        return "provider_transient", True
    if isinstance(exc, ProviderPermanentError):
        return "provider_permanent", False
    return "internal", False


def create_job(
    store: JobStore,
    settings: Settings,
    source: Path,
    title: str | None = None,
    options: JobOptions | None = None,
    job_id: str | None = None,
    filename: str | None = None,
) -> Job:
    """Copy *source* into a fresh job workspace and insert the queued job row. *filename* is the
    name recorded on the job (default: the source file name; the API passes the client's name
    because it spools uploads under a fixed temp name)."""
    source = Path(source)
    job_id = job_id or new_job_id()
    paths = JobPaths(settings.data_dir, job_id)
    paths.job_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, paths.source(source.suffix.lower()))
    filename = filename or source.name
    job = Job(id=job_id, title=title or Path(filename).stem, filename=filename, options=options or JobOptions())
    return store.create_job(job, settings.snapshot())


def _settings_for_job(settings: Settings, store: JobStore, job_id: str) -> Settings:
    snapshot = store.settings_snapshot(job_id)
    if not snapshot:
        return settings
    try:
        return settings.with_snapshot(snapshot)
    except ValueError as exc:
        log.warning("job %s: stored settings snapshot ignored (%s)", job_id, exc)
        return settings


def _stages_to_run(store: JobStore, job_id: str, from_stage: Stage | None) -> tuple[list[Stage], frozenset[Stage]]:
    """``(stages to run, stages forced to regenerate their outputs)``.

    A stage is *forced* when it is being re-run although it ran before: it sits at or after an
    explicit *from_stage*, or its record was reset to pending / left failed after an earlier
    attempt (``store.requeue(from_stage=...)``, PUT /cast, crash recovery). Forced stages must
    not take the "outputs already exist" shortcut, otherwise a retry from ``analyze`` or
    ``cast`` would silently keep the old scripts or cast.
    """
    records = {record.stage: record for record in store.stage_records(job_id)}
    restart_from = STAGE_ORDER.index(from_stage) if from_stage is not None else len(STAGE_ORDER)
    stages: list[Stage] = []
    forced: set[Stage] = set()
    for position, stage in enumerate(STAGE_ORDER):
        record = records.get(stage)
        if record is not None and record.state == StageState.done and position < restart_from:
            continue
        stages.append(stage)
        if position >= restart_from or (record is not None and record.attempts > 0):
            forced.add(stage)
    return stages, frozenset(forced)


def run_job(
    job_id: str,
    settings: Settings,
    store: JobStore,
    providers: Providers | None = None,
    stop_event: threading.Event | None = None,
    from_stage: Stage | str | None = None,
) -> Job:
    """Run every pending stage of *job_id* and return the final job row.

    Stages already ``done`` are skipped unless *from_stage* names them (or an earlier stage).
    The returned job is ``done``, ``failed`` (with a typed error), ``cancelled`` (user request)
    or ``queued`` again (worker shutdown). Raises ``KeyError`` for an unknown job id. The job
    must be ``queued`` (any non-running status when *from_stage* is given): a job that is not
    claimable is returned untouched, so a stale queue entry is a no-op.
    """
    job = store.get_job(job_id)
    if job is None:
        raise KeyError(f"unknown job id {job_id!r}")
    from_stage = Stage(from_stage) if isinstance(from_stage, str) else from_stage
    # Claim the job atomically (queued -> running). A stale queue entry (the job was cancelled,
    # deleted+recreated, retried while still queued, or is already running on another worker)
    # fails the claim and is a no-op. An explicit from_stage is a direct caller's request to
    # re-run a finished job, so any non-running status may be claimed then.
    claimable = (JobStatus.queued,) if from_stage is None else (JobStatus.queued, JobStatus.done, JobStatus.failed, JobStatus.cancelled)
    if not store.claim(job_id, claimable):
        log.info("job %s not claimed: status is %s, not one of %s", job_id, job.status.value, ", ".join(s.value for s in claimable))
        return store.get_job(job_id) or job
    if job.cancel_requested:
        error = JobError(stage=job.stage.value if job.stage else "queued", error_type="cancelled", message="cancelled before it started", retryable=True)
        store.set_status(job_id, JobStatus.cancelled, error=error)
        return store.get_job(job_id) or job

    settings = _settings_for_job(settings, store, job_id)
    paths = JobPaths(settings.data_dir, job_id)
    paths.job_dir.mkdir(parents=True, exist_ok=True)
    ledger = UsageLedger(store, job_id, settings.prices)
    handler, log_filter = attach_job_log(job_id, paths.log, settings.log_level)
    stop = stop_event or threading.Event()
    store.update_job(job_id, started_at=job.started_at or now_iso())
    stage: Stage | None = None
    try:
        with UsageRouter.bind(ledger):
            active = providers or build_providers(settings, ledger)
            stages, forced = _stages_to_run(store, job_id, from_stage)
            ctx = JobContext(
                job=store.get_job(job_id) or job, settings=settings, paths=paths, store=store, providers=active,
                ledger=ledger, cache=ClipCache(paths.cache_root), stop_event=stop, log_filter=log_filter, forced=forced,
            )
            families = ", ".join(f"{cap}={desc['family']}" for cap, desc in active.describe().items())
            ctx.log("info", f"run started (providers: {families})")
            for stage in stages:
                ctx.stage = stage
                store.update_job(job_id, stage=stage)
                store.set_stage(job_id, stage, state=StageState.running, done=0, message="", bump_attempts=True)
                ctx.log("info", f"stage {stage.value} started")
                STAGE_FUNCTIONS[stage](ctx)
                store.set_stage(job_id, stage, state=StageState.done)
                ctx.log("info", f"stage {stage.value} done")
                ctx.job = store.get_job(job_id) or ctx.job
            store.set_status(job_id, JobStatus.done)
            ctx.log("info", "run finished: done")
    except JobCancelled as exc:
        stage_name = stage.value if stage else "queued"
        if stage is not None:
            store.set_stage(job_id, stage, state=StageState.pending)
        if exc.reason == "shutdown":
            store.set_status(job_id, JobStatus.queued)
            _event(store, job_id, "warn", stage, "worker shutting down; job returned to the queue")
        else:
            error = JobError(stage=stage_name, error_type="cancelled", message=str(exc), retryable=True, unit=exc.unit)
            store.set_status(job_id, JobStatus.cancelled, error=error)
            _event(store, job_id, "warn", stage, "cancelled by user request")
    except Exception as exc:  # noqa: BLE001 - every failure becomes a typed JobError
        error_type, retryable = classify(exc)
        message = scrub(f"{type(exc).__name__}: {exc}", settings.secrets)
        error = JobError(
            stage=stage.value if stage else "startup",
            error_type=error_type,  # type: ignore[arg-type]
            message=message,
            retryable=retryable,
            unit=getattr(exc, "unit", None),
        )
        if stage is not None:
            store.set_stage(job_id, stage, state=StageState.failed, message=message[:MAX_STAGE_MESSAGE])
        store.set_status(job_id, JobStatus.failed, error=error)
        _event(store, job_id, "error", stage, f"{error_type}: {message}" + (f" (unit {error.unit})" if error.unit else ""))
        if error_type == "internal":
            job_logger.error("[%s] internal error\n%s", job_id, scrub(traceback.format_exc(), settings.secrets), extra={"job_id": job_id})
    finally:
        detach_job_log(handler)
    return store.get_job(job_id) or job


def _event(store: JobStore, job_id: str, level: str, stage: Stage | None, message: str) -> None:
    job_logger.log(logging.WARNING if level != "error" else logging.ERROR, "[%s] %s", job_id, message, extra={"job_id": job_id})
    store.append_event(job_id, level, stage.value if stage else None, message)
