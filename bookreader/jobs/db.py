"""bookreader.jobs.db - JobStore, the SQLite-backed job state (jobs, stages, events, usage).

One connection per thread (``threading.local``), WAL journal, ``synchronous=NORMAL``, a 5 s busy
timeout and foreign keys on. Every write is a short autocommit transaction (multi-row writes use
one ``BEGIN IMMEDIATE`` block), so API reads and worker writes interleave without long locks.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bookreader.types import (
    STAGE_ORDER,
    Job,
    JobError,
    JobEvent,
    JobOptions,
    JobStatus,
    Stage,
    StageRecord,
    StageState,
    UsageEvent,
    UsageSummary,
)

log = logging.getLogger(__name__)

BUSY_TIMEOUT_MS = 5000
TERMINAL_STATUSES: frozenset[JobStatus] = frozenset({JobStatus.done, JobStatus.failed, JobStatus.cancelled})

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    filename TEXT NOT NULL,
    source_sha256 TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    stage TEXT,
    options_json TEXT NOT NULL,
    settings_json TEXT NOT NULL,
    error_json TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS job_stages (
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending',
    done INTEGER NOT NULL DEFAULT 0,
    total INTEGER NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    attempts INTEGER NOT NULL DEFAULT 0,
    started_at TEXT,
    finished_at TEXT,
    PRIMARY KEY (job_id, stage)
);
CREATE TABLE IF NOT EXISTS job_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    ts TEXT NOT NULL,
    level TEXT NOT NULL,
    stage TEXT,
    message TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS job_events_job ON job_events(job_id, id);
CREATE TABLE IF NOT EXISTS usage_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    ts TEXT NOT NULL,
    capability TEXT NOT NULL,
    family TEXT NOT NULL,
    unit_type TEXT NOT NULL,
    units REAL NOT NULL,
    cache_hit INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    meta_json TEXT
);
CREATE INDEX IF NOT EXISTS usage_events_job ON usage_events(job_id);
"""

_JOB_COLUMNS: dict[str, str] = {
    "title": "title",
    "filename": "filename",
    "source_sha256": "source_sha256",
    "status": "status",
    "stage": "stage",
    "options": "options_json",
    "error": "error_json",
    "cancel_requested": "cancel_requested",
    "started_at": "started_at",
    "finished_at": "finished_at",
}


def now_iso() -> str:
    """Current UTC time as an ISO-8601 string with millisecond precision."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_job_id() -> str:
    """A short random job id (12 hex characters)."""
    return uuid.uuid4().hex[:12]


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _encode_field(name: str, value: Any) -> Any:
    if name == "options":
        if isinstance(value, JobOptions):
            return value.model_dump_json()
        return JobOptions.model_validate(value or {}).model_dump_json()
    if name == "error":
        if value is None:
            return None
        return value.model_dump_json() if isinstance(value, JobError) else JobError.model_validate(value).model_dump_json()
    if name == "cancel_requested":
        return 1 if value else 0
    if name in ("status", "stage"):
        return _enum_value(value)
    return value


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"],
        title=row["title"],
        filename=row["filename"],
        source_sha256=row["source_sha256"],
        status=JobStatus(row["status"]),
        stage=Stage(row["stage"]) if row["stage"] else None,
        options=JobOptions.model_validate_json(row["options_json"]),
        error=JobError.model_validate_json(row["error_json"]) if row["error_json"] else None,
        cancel_requested=bool(row["cancel_requested"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
    )


def _row_to_stage(row: sqlite3.Row) -> StageRecord:
    return StageRecord(
        stage=Stage(row["stage"]),
        state=StageState(row["state"]),
        done=int(row["done"]),
        total=int(row["total"]),
        message=row["message"],
        attempts=int(row["attempts"]),
        started_at=row["started_at"],
        finished_at=row["finished_at"],
    )


class JobStore:
    """SQLite job state. Safe to share between threads; each thread gets its own connection."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._conn().executescript(SCHEMA)

    # ------------------------------------------------------------------ connections
    def _conn(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.path), timeout=BUSY_TIMEOUT_MS / 1000, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        """Close the calling thread's connection (other threads keep theirs)."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def _transaction(self) -> "_Transaction":
        return _Transaction(self._conn())

    # ------------------------------------------------------------------ jobs
    def create_job(self, job: Job, settings_snapshot: str) -> Job:
        """Insert *job* (status queued unless set) with its five pending stage rows; returns the stored job."""
        ts = now_iso()
        created_at = job.created_at or ts
        with self._transaction() as conn:
            conn.execute(
                "INSERT INTO jobs (id, title, filename, source_sha256, status, stage, options_json, settings_json, "
                "error_json, cancel_requested, created_at, updated_at, started_at, finished_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job.id, job.title, job.filename, job.source_sha256, job.status.value,
                    job.stage.value if job.stage else None, job.options.model_dump_json(), settings_snapshot,
                    job.error.model_dump_json() if job.error else None, 1 if job.cancel_requested else 0,
                    created_at, ts, job.started_at, job.finished_at,
                ),
            )
            conn.executemany(
                "INSERT INTO job_stages (job_id, stage) VALUES (?, ?)",
                [(job.id, stage.value) for stage in STAGE_ORDER],
            )
        stored = self.get_job(job.id)
        assert stored is not None
        return stored

    def get_job(self, job_id: str) -> Job | None:
        """The job row as a :class:`Job`, or None when unknown."""
        row = self._conn().execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _row_to_job(row) if row else None

    def list_jobs(self, limit: int = 100) -> list[Job]:
        """Newest first."""
        rows = self._conn().execute("SELECT * FROM jobs ORDER BY created_at DESC, rowid DESC LIMIT ?", (int(limit),)).fetchall()
        return [_row_to_job(row) for row in rows]

    def settings_snapshot(self, job_id: str) -> str | None:
        """The ``Settings.snapshot()`` JSON stored when the job was created."""
        row = self._conn().execute("SELECT settings_json FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return row["settings_json"] if row else None

    def update_job(self, job_id: str, **fields: Any) -> None:
        """Set any of title, filename, source_sha256, status, stage, options, error, cancel_requested,
        started_at, finished_at; ``updated_at`` is always refreshed."""
        unknown = [name for name in fields if name not in _JOB_COLUMNS]
        if unknown:
            raise ValueError(f"unknown job field(s): {', '.join(unknown)}")
        assignments = ["updated_at = ?"]
        values: list[Any] = [now_iso()]
        for name, value in fields.items():
            assignments.append(f"{_JOB_COLUMNS[name]} = ?")
            values.append(_encode_field(name, value))
        values.append(job_id)
        self._conn().execute(f"UPDATE jobs SET {', '.join(assignments)} WHERE id = ?", values)

    def set_status(self, job_id: str, status: JobStatus | str, stage: Stage | str | None = None, error: JobError | None = None) -> None:
        """Set the status (and optionally the current stage); *error* replaces the stored error
        (None clears it). Terminal statuses stamp ``finished_at``; others clear it."""
        status = JobStatus(_enum_value(status))
        fields: dict[str, Any] = {"status": status, "error": error}
        if stage is not None:
            fields["stage"] = Stage(_enum_value(stage))
        fields["finished_at"] = now_iso() if status in TERMINAL_STATUSES else None
        self.update_job(job_id, **fields)

    def claim(self, job_id: str, from_statuses: tuple[JobStatus, ...] = (JobStatus.queued,)) -> bool:
        """Atomically move the job to ``running`` when its status is one of *from_statuses*.

        Returns False (and changes nothing) when the row is missing or already in another
        status, so two workers holding the same id, or a stale queue entry left behind by a
        cancel / delete / retry sequence, can never run one job twice.
        """
        statuses = [JobStatus(_enum_value(status)).value for status in from_statuses] or [JobStatus.queued.value]
        placeholders = ", ".join("?" for _ in statuses)
        cursor = self._conn().execute(
            f"UPDATE jobs SET status = ?, error_json = NULL, finished_at = NULL, updated_at = ? "
            f"WHERE id = ? AND status IN ({placeholders})",
            (JobStatus.running.value, now_iso(), job_id, *statuses),
        )
        return cursor.rowcount == 1

    def request_cancel(self, job_id: str) -> None:
        """Set the cancel flag; the running job stops at its next checkpoint."""
        self.update_job(job_id, cancel_requested=True)

    def cancel_requested(self, job_id: str) -> bool:
        """Whether cancellation was requested (False for unknown jobs)."""
        row = self._conn().execute("SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return bool(row and row["cancel_requested"])

    def requeue(self, job_id: str, from_stage: Stage | str | None = None) -> None:
        """Prepare a retry: status queued, cancel flag and error cleared, and (when *from_stage* is
        given) every stage from it onward reset to pending."""
        self.update_job(job_id, status=JobStatus.queued, cancel_requested=False, error=None, finished_at=None)
        if from_stage is not None:
            self.reset_stages_from(job_id, from_stage)

    def requeue_orphans(self) -> list[str]:
        """Jobs left ``running`` by a crashed process go back to ``queued``; returns their ids."""
        conn = self._conn()
        rows = conn.execute("SELECT id FROM jobs WHERE status = ?", (JobStatus.running.value,)).fetchall()
        ids = [row["id"] for row in rows]
        if not ids:
            return []
        ts = now_iso()
        with self._transaction() as conn:
            for job_id in ids:
                conn.execute("UPDATE jobs SET status = ?, updated_at = ? WHERE id = ?", (JobStatus.queued.value, ts, job_id))
                conn.execute(
                    "UPDATE job_stages SET state = ?, finished_at = NULL WHERE job_id = ? AND state = ?",
                    (StageState.pending.value, job_id, StageState.running.value),
                )
        log.info("requeued %d orphaned job(s): %s", len(ids), ", ".join(ids))
        return ids

    def delete_job(self, job_id: str) -> bool:
        """Delete the job and, through cascades, its stages, events and usage rows."""
        cursor = self._conn().execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        return cursor.rowcount > 0

    # ------------------------------------------------------------------ stages
    def stage_records(self, job_id: str) -> list[StageRecord]:
        """The job's stage rows in pipeline order."""
        rows = self._conn().execute("SELECT * FROM job_stages WHERE job_id = ?", (job_id,)).fetchall()
        by_stage = {row["stage"]: _row_to_stage(row) for row in rows}
        return [by_stage[stage.value] for stage in STAGE_ORDER if stage.value in by_stage]

    def set_stage(
        self,
        job_id: str,
        stage: Stage | str,
        state: StageState | str | None = None,
        done: int | None = None,
        total: int | None = None,
        message: str | None = None,
        bump_attempts: bool = False,
    ) -> None:
        """Update one stage row. ``running`` stamps ``started_at`` (and clears ``finished_at``);
        ``done``/``failed`` stamp ``finished_at``; ``pending`` clears both."""
        assignments: list[str] = []
        values: list[Any] = []
        if state is not None:
            state = StageState(_enum_value(state))
            assignments.append("state = ?")
            values.append(state.value)
            ts = now_iso()
            if state == StageState.running:
                assignments.extend(["started_at = ?", "finished_at = NULL"])
                values.append(ts)
            elif state in (StageState.done, StageState.failed):
                assignments.append("finished_at = ?")
                values.append(ts)
            else:
                assignments.extend(["started_at = NULL", "finished_at = NULL"])
        for column, value in (("done", done), ("total", total), ("message", message)):
            if value is not None:
                assignments.append(f"{column} = ?")
                values.append(value)
        if bump_attempts:
            assignments.append("attempts = attempts + 1")
        if not assignments:
            return
        values.extend([job_id, _enum_value(stage)])
        self._conn().execute(f"UPDATE job_stages SET {', '.join(assignments)} WHERE job_id = ? AND stage = ?", values)

    def reset_stages_from(self, job_id: str, stage: Stage | str) -> None:
        """Set *stage* and every later stage back to pending with zeroed progress (attempts are kept)."""
        start = STAGE_ORDER.index(Stage(_enum_value(stage)))
        with self._transaction() as conn:
            for later in STAGE_ORDER[start:]:
                conn.execute(
                    "UPDATE job_stages SET state = ?, done = 0, total = 0, message = '', started_at = NULL, finished_at = NULL "
                    "WHERE job_id = ? AND stage = ?",
                    (StageState.pending.value, job_id, later.value),
                )

    # ------------------------------------------------------------------ events
    def append_event(self, job_id: str, level: str, stage: str | None, message: str) -> int:
        """Append a job event (level: info | warn | error) and return its id."""
        cursor = self._conn().execute(
            "INSERT INTO job_events (job_id, ts, level, stage, message) VALUES (?, ?, ?, ?, ?)",
            (job_id, now_iso(), level, _enum_value(stage) if stage is not None else None, message),
        )
        return int(cursor.lastrowid or 0)

    def events_after(self, job_id: str, after: int = 0, limit: int = 200) -> list[JobEvent]:
        """Up to *limit* events with ids greater than *after*, oldest first."""
        rows = self._conn().execute(
            "SELECT * FROM job_events WHERE job_id = ? AND id > ? ORDER BY id LIMIT ?",
            (job_id, int(after), int(limit)),
        ).fetchall()
        return [
            JobEvent(id=row["id"], job_id=row["job_id"], ts=row["ts"], level=row["level"], stage=row["stage"], message=row["message"])
            for row in rows
        ]

    # ------------------------------------------------------------------ usage
    def add_usage(self, job_id: str, event: UsageEvent) -> int:
        """Store one usage row and return its id."""
        cursor = self._conn().execute(
            "INSERT INTO usage_events (job_id, ts, capability, family, unit_type, units, cache_hit, cost_usd, duration_ms, meta_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                job_id, now_iso(), event.capability, event.family, event.unit_type, float(event.units),
                1 if event.cache_hit else 0, float(event.cost_usd), int(event.duration_ms),
                json.dumps(event.meta, sort_keys=True) if event.meta else None,
            ),
        )
        return int(cursor.lastrowid or 0)

    def usage_summary(self, job_id: str) -> UsageSummary:
        """``calls`` counts recorded provider calls (rows that were not cache hits), ``cache_hits``
        the rows recorded for cached work, ``cost_usd`` the priced total and ``by_capability``
        the summed units per capability and unit type (hits included)."""
        conn = self._conn()
        totals = conn.execute(
            "SELECT SUM(CASE WHEN cache_hit = 0 THEN 1 ELSE 0 END) AS calls, "
            "SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) AS hits, SUM(cost_usd) AS cost "
            "FROM usage_events WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        rows = conn.execute(
            "SELECT capability, unit_type, SUM(units) AS units FROM usage_events WHERE job_id = ? "
            "GROUP BY capability, unit_type ORDER BY capability, unit_type",
            (job_id,),
        ).fetchall()
        by_capability: dict[str, dict[str, float]] = {}
        for row in rows:
            by_capability.setdefault(row["capability"], {})[row["unit_type"]] = float(row["units"])
        return UsageSummary(
            calls=int(totals["calls"] or 0),
            cache_hits=int(totals["hits"] or 0),
            cost_usd=round(float(totals["cost"] or 0.0), 6),
            by_capability=by_capability,
        )

    def usage_count(self, job_id: str) -> int:
        """Number of usage rows recorded for the job."""
        row = self._conn().execute("SELECT COUNT(*) AS n FROM usage_events WHERE job_id = ?", (job_id,)).fetchone()
        return int(row["n"])


class _Transaction:
    """``BEGIN IMMEDIATE`` ... ``COMMIT`` (or ``ROLLBACK`` on error) on an autocommit connection."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def __enter__(self) -> sqlite3.Connection:
        self.conn.execute("BEGIN IMMEDIATE")
        return self.conn

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any) -> None:
        if exc_type is None:
            self.conn.execute("COMMIT")
        else:
            self.conn.execute("ROLLBACK")

