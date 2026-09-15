"""bookreader.api.schemas - request and response models of the HTTP API.

Every body serialized over HTTP is a pydantic model with ``extra="forbid"`` (``StrictModel``),
reusing the domain models from :mod:`bookreader.types` wherever the API exposes them unchanged
(``StageRecord``, ``JobError``, ``JobOptions``, ``Estimate``, ``Cast``, ``VoiceInfo``,
``JobEvent``, ``JobManifest``).
"""
from __future__ import annotations

from typing import Any

from pydantic import Field

from bookreader.types import (
    Cast,
    Estimate,
    JobError,
    JobEvent,
    JobOptions,
    Stage,
    StageRecord,
    StrictModel,
    VoiceInfo,
)


class JobCreated(StrictModel):
    """``POST /api/jobs`` response."""

    job_id: str
    status: str
    status_url: str


class JobListItem(StrictModel):
    """One row of ``GET /api/jobs``."""

    id: str
    title: str
    status: str
    stage: str | None
    overall_pct: float
    created_at: str
    updated_at: str


class JobList(StrictModel):
    jobs: list[JobListItem]


class UsageBrief(StrictModel):
    """The three usage numbers shown on the job panel."""

    calls: int = 0
    cache_hits: int = 0
    cost_usd: float = 0.0


class ChapterBrief(StrictModel):
    """Chapter readiness as derived from the job directory."""

    index: int
    title: str
    ready: bool
    duration_ms: int | None = None


class JobStatusOut(StrictModel):
    """``GET /api/jobs/{job_id}`` response."""

    id: str
    title: str
    filename: str
    status: str
    stage: str | None
    overall_pct: float
    stages: list[StageRecord]
    error: JobError | None
    options: JobOptions
    estimate: Estimate | None
    usage: UsageBrief
    chapters: list[ChapterBrief]
    providers: dict[str, dict[str, str]]
    created_at: str
    updated_at: str
    started_at: str | None
    finished_at: str | None


class EventsOut(StrictModel):
    events: list[JobEvent]
    last_id: int


class CastOut(StrictModel):
    cast: Cast
    voices: list[VoiceInfo]


class CastUpdate(StrictModel):
    """``PUT /api/jobs/{job_id}/cast`` body: character name (or ``NARRATOR``) -> voice id."""

    overrides: dict[str, str] = Field(default_factory=dict)


class JobAction(StrictModel):
    """Response of retry / cancel / re-cast."""

    job_id: str
    status: str


class RetryRequest(StrictModel):
    from_stage: Stage | None = None


class UsageOut(StrictModel):
    calls: int
    cache_hits: int
    cost_usd: float
    by_capability: dict[str, dict[str, float]]
    events_count: int


class ArtifactFile(StrictModel):
    path: str
    bytes: int
    url: str


class ArtifactList(StrictModel):
    files: list[ArtifactFile]


class QueueOut(StrictModel):
    backend: str
    depth: int
    running: list[str]


class HealthOut(StrictModel):
    ok: bool
    version: str
    providers: list[dict[str, Any]]
    ffmpeg: bool
    queue: QueueOut
    data_dir_writable: bool
