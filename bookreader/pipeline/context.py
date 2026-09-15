"""bookreader.pipeline.context - JobContext, everything a stage function needs for one job.

Besides the job, settings, paths, store, providers, ledger and cache, the context offers:

* ``progress(substage, done, total, message)`` - writes the stage record (throttled to one
  write per 250 ms, the final value always written) and appends an info event whenever the
  substage changes;
* ``log(level, message)`` - to the job logger (and so to ``job.log``) and to ``job_events``;
* ``check_cancelled()`` - raises ``JobCancelled('shutdown')`` when the worker's stop event is
  set and ``JobCancelled('user')`` when the store's cancel flag is set (polled at most every
  250 ms unless ``force=True``);
* ``substage(name)`` - a context manager that names and times a substage in the log.
"""
from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from bookreader.jobs.db import JobStore
from bookreader.jobs.paths import JobPaths
from bookreader.pipeline.cache import ClipCache
from bookreader.providers.base import Providers
from bookreader.settings import Settings
from bookreader.types import Job, JobCancelled, Stage
from bookreader.usage import UsageLedger

PROGRESS_INTERVAL_S = 0.25
CANCEL_POLL_INTERVAL_S = 0.25
LOG_LEVELS: dict[str, int] = {"info": logging.INFO, "warn": logging.WARNING, "error": logging.ERROR}
JOB_LOGGER_NAME = "bookreader.job"
PACKAGE_LOGGER_NAME = "bookreader"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

job_logger = logging.getLogger(JOB_LOGGER_NAME)
job_logger.setLevel(logging.INFO)          # ctx.log must reach job.log whatever the root level is


class JobLogFilter(logging.Filter):
    """Passes records tagged with this job's id (``extra={"job_id": ...}``) or emitted from one
    of the job's threads (registered through :meth:`JobContext.register_thread`)."""

    def __init__(self, job_id: str) -> None:
        super().__init__(name="")
        self.job_id = job_id
        self.threads: set[int] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(record, "job_id", None) == self.job_id or record.thread in self.threads


def attach_job_log(job_id: str, path: Path, level: str = "INFO") -> tuple[logging.Handler, JobLogFilter]:
    """Attach a ``job.log`` file handler (with a job filter) to the package logger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    numeric = getattr(logging, level.upper(), logging.INFO)
    handler.setLevel(numeric)
    job_logger.setLevel(min(logging.INFO, numeric))
    log_filter = JobLogFilter(job_id)
    log_filter.threads.add(threading.get_ident())
    handler.addFilter(log_filter)
    logging.getLogger(PACKAGE_LOGGER_NAME).addHandler(handler)
    return handler, log_filter


def detach_job_log(handler: logging.Handler) -> None:
    logging.getLogger(PACKAGE_LOGGER_NAME).removeHandler(handler)
    handler.close()


@dataclass
class JobContext:
    """Shared state for the stage functions of one job run."""

    job: Job
    settings: Settings
    paths: JobPaths
    store: JobStore
    providers: Providers
    ledger: UsageLedger
    cache: ClipCache
    stop_event: threading.Event = field(default_factory=threading.Event)
    stage: Stage | None = None
    log_filter: JobLogFilter | None = None
    _substage: str | None = field(default=None, init=False, repr=False)
    _last_progress_write: float = field(default=0.0, init=False, repr=False)
    _last_cancel_poll: float = field(default=0.0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _produced: set[str] = field(default_factory=set, init=False, repr=False)

    # ------------------------------------------------------------------ logging
    def log(self, level: str, message: str) -> None:
        """Record *message* in ``job.log`` and ``job_events`` (level: info | warn | error)."""
        stage = self.stage.value if self.stage else None
        job_logger.log(LOG_LEVELS.get(level, logging.INFO), "[%s] %s", self.job.id, message, extra={"job_id": self.job.id})
        self.store.append_event(self.job.id, level, stage, message)

    def register_thread(self) -> None:
        """Make the calling thread's log records part of this job's ``job.log`` (executor initializer)."""
        if self.log_filter is not None:
            self.log_filter.threads.add(threading.get_ident())

    # ------------------------------------------------------------------ progress
    def progress(self, substage: str, done: int, total: int, message: str = "") -> None:
        """Update the current stage's record; throttled, but the final value is always written."""
        if self.stage is None:
            return
        with self._lock:
            now = time.monotonic()
            transition = substage != self._substage
            if transition:
                self._substage = substage
            final = total > 0 and done >= total
            if not (transition or final or now - self._last_progress_write >= PROGRESS_INTERVAL_S):
                return
            self._last_progress_write = now
        self.store.set_stage(self.job.id, self.stage, done=done, total=total, message=message)
        if transition:
            self.log("info", f"{self.stage.value}/{substage}: {message or f'{done}/{total}'}")

    @contextmanager
    def substage(self, name: str) -> Iterator[None]:
        """Name and time a substage; the timing goes to the job log at debug level."""
        started = time.monotonic()
        job_logger.debug("[%s] %s: %s started", self.job.id, self.stage.value if self.stage else "-", name, extra={"job_id": self.job.id})
        yield
        elapsed = (time.monotonic() - started) * 1000
        job_logger.debug("[%s] %s: %s finished in %.0f ms", self.job.id, self.stage.value if self.stage else "-", name, elapsed, extra={"job_id": self.job.id})

    # ------------------------------------------------------------------ cache bookkeeping
    def mark_produced(self, kind: str, key: str) -> None:
        """Remember that this run created the cache entry, so reusing it later in the same run
        (a bed shared by two chapters) is not reported as a cache hit."""
        with self._lock:
            self._produced.add(f"{kind}/{key}")

    def produced(self, kind: str, key: str) -> bool:
        with self._lock:
            return f"{kind}/{key}" in self._produced

    # ------------------------------------------------------------------ cancellation
    def check_cancelled(self, force: bool = False) -> None:
        """Raise :class:`JobCancelled` when the worker is stopping or the user asked to cancel."""
        if self.stop_event.is_set():
            raise JobCancelled("shutdown")
        now = time.monotonic()
        with self._lock:
            due = force or now - self._last_cancel_poll >= CANCEL_POLL_INTERVAL_S
            if due:
                self._last_cancel_poll = now
        if due and self.store.cancel_requested(self.job.id):
            raise JobCancelled("user")
