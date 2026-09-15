"""bookreader.api.routes - every ``/api`` endpoint.

The router reads process state (settings, store, providers, queue) from ``app.state`` through the
:func:`deps` dependency. Job bookkeeping lives in :class:`bookreader.jobs.db.JobStore`; the
endpoints only translate HTTP into store calls, queue submissions and files under the job
directory. Artifact downloads are guarded by resolving the requested path and checking that it
stays inside the job directory.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from bookreader import __version__
from bookreader.api.schemas import (
    ArtifactFile,
    ArtifactList,
    CastOut,
    CastUpdate,
    ChapterBrief,
    EventsOut,
    HealthOut,
    JobAction,
    JobCreated,
    JobList,
    JobListItem,
    JobStatusOut,
    QueueOut,
    RetryRequest,
    UsageBrief,
    UsageOut,
)
from bookreader.audio.export import ffmpeg_path
from bookreader.ingest import SUPPORTED_SUFFIXES
from bookreader.jobs.db import TERMINAL_STATUSES, JobStore
from bookreader.jobs.paths import JobPaths, read_json, write_json
from bookreader.jobs.queue import JobQueue
from bookreader.manifest import build_job_manifest
from bookreader.pipeline.run import create_job
from bookreader.providers.base import Providers, describe_providers
from bookreader.settings import Settings
from bookreader.types import (
    NARRATOR,
    STAGE_WEIGHTS,
    Book,
    Cast,
    CastBible,
    ChapterManifest,
    Estimate,
    Job,
    JobError,
    JobManifest,
    JobOptions,
    JobStatus,
    Stage,
    StageRecord,
    StageState,
    VoiceInfo,
    normalize_name,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

MB = 1024 * 1024
UPLOAD_CHUNK_BYTES = MB
MAX_EVENT_LIMIT = 1000
MAX_LOG_LINES = 10_000
MAX_SQLITE_INT = 2**63 - 1
MAX_FILENAME_CHARS = 200
MEDIA_TYPES: dict[str, str] = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".json": "application/json",
    ".log": "text/plain; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".key": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
}
DEFAULT_MEDIA_TYPE = "application/octet-stream"


# --------------------------------------------------------------------------- dependencies
@dataclass
class ApiDeps:
    """The process-wide objects the endpoints work with (populated by the app lifespan)."""

    settings: Settings
    store: JobStore
    providers: Providers
    queue: JobQueue


def deps(request: Request) -> ApiDeps:
    state = request.app.state
    return ApiDeps(settings=state.settings, store=state.store, providers=state.providers, queue=state.queue)


def _root(request: Request) -> str:
    return str(request.scope.get("root_path", "")).rstrip("/")


def _load_job(store: JobStore, job_id: str) -> Job:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id!r}")
    return job


def _require_not_running(job: Job) -> None:
    if job.status == JobStatus.running:
        raise HTTPException(status_code=409, detail=f"job {job.id} is running")


def _paths(settings: Settings, job_id: str) -> JobPaths:
    return JobPaths(settings.data_dir, job_id)


def _discard_from_queue(queue: JobQueue, job_id: str) -> None:
    """Drop a waiting id from queue backends that support it (a cancelled or deleted job must not
    be picked up by a worker later)."""
    discard = getattr(queue, "discard", None)
    if callable(discard):
        discard(job_id)


def _hidden_artifact(paths: JobPaths, path: Path) -> bool:
    """The uploaded source and dot-files (atomic-write temp files, health probes) are not artifacts."""
    return path.name.startswith(".") or (path.parent == paths.job_dir and path.stem == "source")


# --------------------------------------------------------------------------- derived views
def overall_pct(records: list[StageRecord]) -> float:
    """Weighted progress: done stages count fully, the running stage by ``done/total``."""
    pct = 0.0
    for record in records:
        weight = STAGE_WEIGHTS[record.stage]
        if record.state == StageState.done:
            pct += weight
        elif record.state == StageState.running and record.total > 0:
            pct += weight * min(1.0, record.done / record.total)
    return round(min(100.0, pct), 1)


def _chapters(paths: JobPaths) -> list[ChapterBrief]:
    """Chapter readiness from ``book.json`` plus the rendered chapter manifests."""
    if not paths.book.is_file():
        return []
    book = read_json(paths.book)
    out: list[ChapterBrief] = []
    for entry in book.get("chapters", []):
        index = int(entry["index"])
        ready = paths.mix(index).is_file()
        duration: int | None = None
        manifest_path = paths.chapter_manifest(index)
        if ready and manifest_path.is_file():
            duration = int(read_json(manifest_path).get("duration_ms", 0))
        out.append(ChapterBrief(index=index, title=str(entry.get("title", "")), ready=ready, duration_ms=duration))
    return out


def _estimate(paths: JobPaths) -> Estimate | None:
    if not paths.estimate.is_file():
        return None
    return Estimate.model_validate(read_json(paths.estimate))


def _status_out(job: Job, d: ApiDeps) -> JobStatusOut:
    paths = _paths(d.settings, job.id)
    records = d.store.stage_records(job.id)
    summary = d.store.usage_summary(job.id)
    return JobStatusOut(
        id=job.id,
        title=job.title,
        filename=job.filename,
        status=job.status.value,
        stage=job.stage.value if job.stage else None,
        overall_pct=overall_pct(records),
        stages=records,
        error=job.error,
        options=job.options,
        estimate=_estimate(paths),
        usage=UsageBrief(calls=summary.calls, cache_hits=summary.cache_hits, cost_usd=summary.cost_usd),
        chapters=_chapters(paths),
        providers=d.providers.describe(),
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


def _voices(paths: JobPaths) -> list[VoiceInfo]:
    if not paths.voices.is_file():
        return []
    return [VoiceInfo.model_validate(item) for item in read_json(paths.voices)]


def _known_character(name: str, cast: Cast | None, bible: CastBible | None) -> bool:
    key = normalize_name(name)
    if key == NARRATOR.lower():
        return True
    if cast is not None and any(normalize_name(assignment.character) == key for assignment in cast.characters):
        return True
    return bible is not None and bible.find(name) is not None


def _clear_render_outputs(paths: JobPaths) -> None:
    """Remove everything produced after the TTS substage so a re-cast re-renders every chapter
    while ``tts.json`` (and the shared clip cache) keep unchanged voices free."""
    if paths.chapters_dir.is_dir():
        for chapter_dir in paths.chapters_dir.iterdir():
            if not chapter_dir.is_dir():
                continue
            for entry in chapter_dir.iterdir():
                if entry.is_file() and entry.name != "tts.json":
                    entry.unlink()
    paths.manifest.unlink(missing_ok=True)


def _normalized_options(options: JobOptions) -> JobOptions:
    """Reject chapter numbers below 1 (400 at upload rather than an input error minutes later);
    duplicates are dropped, an empty list means every chapter."""
    if options.chapters is None:
        return options
    bad = [n for n in options.chapters if n < 1]
    if bad:
        raise HTTPException(status_code=400, detail=f"invalid options: chapters must be >= 1, got {bad[0]}")
    return options.model_copy(update={"chapters": sorted(set(options.chapters)) or None})


def _media_type(path: Path) -> str:
    return MEDIA_TYPES.get(path.suffix.lower(), DEFAULT_MEDIA_TYPE)


# --------------------------------------------------------------------------- jobs
@router.post("/jobs", status_code=202, response_model=JobCreated)
async def submit_job(
    request: Request,
    file: UploadFile = File(...),
    title: str | None = Form(None),
    options: str | None = Form(None),
    d: ApiDeps = Depends(deps),
) -> JobCreated:
    """Upload a book and queue it. 415 for an unsupported extension, 413 when too large, 400 for bad options."""
    filename = Path(file.filename or "").name.strip() or "upload"
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"unsupported file type {suffix or filename!r}; supported: {', '.join(SUPPORTED_SUFFIXES)}")
    if len(filename) > MAX_FILENAME_CHARS:           # keep the display name sane; the disk name never uses it
        filename = filename[: MAX_FILENAME_CHARS - len(suffix)].rstrip() + suffix
    job_options = JobOptions()
    if options:
        try:
            job_options = JobOptions.model_validate_json(options)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=f"invalid options: {exc.errors()[0]['msg']}") from exc
        job_options = _normalized_options(job_options)
    limit = d.settings.max_upload_mb * MB
    with tempfile.TemporaryDirectory(prefix="bookreader-upload-") as tmp:
        target = Path(tmp) / f"upload{suffix}"         # create_job only needs the suffix; a 300-char client name would not open
        received = 0
        with target.open("wb") as fh:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                received += len(chunk)
                if received > limit:
                    raise HTTPException(status_code=413, detail=f"upload exceeds {d.settings.max_upload_mb} MB (BOOKREADER_MAX_UPLOAD_MB)")
                fh.write(chunk)
        job = await run_in_threadpool(create_job, d.store, d.settings, target, (title or "").strip() or None, job_options, filename=filename)
    log.info("job %s created from %s (%d bytes)", job.id, filename, received)
    await run_in_threadpool(d.queue.submit, job.id)
    return JobCreated(job_id=job.id, status=job.status.value, status_url=f"{_root(request)}/api/jobs/{job.id}")


@router.get("/jobs", response_model=JobList)
def list_jobs(d: ApiDeps = Depends(deps)) -> JobList:
    """Every job, newest first."""
    items = [
        JobListItem(
            id=job.id, title=job.title, status=job.status.value, stage=job.stage.value if job.stage else None,
            overall_pct=overall_pct(d.store.stage_records(job.id)), created_at=job.created_at, updated_at=job.updated_at,
        )
        for job in d.store.list_jobs()
    ]
    return JobList(jobs=items)


@router.get("/jobs/{job_id}", response_model=JobStatusOut)
def job_status(job_id: str, d: ApiDeps = Depends(deps)) -> JobStatusOut:
    """Full status: stages, error, estimate, usage, chapter readiness and the providers in use."""
    return _status_out(_load_job(d.store, job_id), d)


@router.get("/jobs/{job_id}/events", response_model=EventsOut)
def job_events(
    job_id: str,
    after: int = Query(0, ge=0, le=MAX_SQLITE_INT),
    limit: int = Query(200, ge=1, le=MAX_EVENT_LIMIT),
    d: ApiDeps = Depends(deps),
) -> EventsOut:
    """Events with ids greater than *after* (oldest first); poll with ``after=last_id``."""
    _load_job(d.store, job_id)
    events = d.store.events_after(job_id, after, limit)
    return EventsOut(events=events, last_id=events[-1].id if events else after)


@router.get("/jobs/{job_id}/log", response_class=PlainTextResponse)
def job_log(job_id: str, lines: int = Query(500, ge=1, le=MAX_LOG_LINES), d: ApiDeps = Depends(deps)) -> PlainTextResponse:
    """The last *lines* lines of ``job.log`` (empty until the job has started)."""
    _load_job(d.store, job_id)
    path = _paths(d.settings, job_id).log
    if not path.is_file():
        return PlainTextResponse("")
    tail: deque[str] = deque(maxlen=lines)
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        tail.extend(fh)
    return PlainTextResponse("".join(tail))


# --------------------------------------------------------------------------- cast
@router.get("/jobs/{job_id}/cast", response_model=CastOut)
def get_cast(job_id: str, d: ApiDeps = Depends(deps)) -> CastOut:
    """The voice assignments plus the catalog they were chosen from; 404 until the cast stage ran."""
    _load_job(d.store, job_id)
    paths = _paths(d.settings, job_id)
    if not paths.cast.is_file():
        raise HTTPException(status_code=404, detail="cast not available yet")
    return CastOut(cast=Cast.model_validate(read_json(paths.cast)), voices=_voices(paths))


@router.put("/jobs/{job_id}/cast", status_code=202, response_model=JobAction)
def update_cast(job_id: str, body: CastUpdate, d: ApiDeps = Depends(deps)) -> JobAction:
    """Pin voices by character name and re-run cast/render/finalize (only changed voices re-synthesize)."""
    job = _load_job(d.store, job_id)
    _require_not_running(job)
    if not body.overrides:
        raise HTTPException(status_code=400, detail="overrides must name at least one character")
    paths = _paths(d.settings, job_id)
    # voices.json and bible.json exist as soon as the cast stage fetched the catalog, so a job whose
    # cast failed on a bad options.cast_overrides entry can still be repaired here (no cast.json yet).
    if not (paths.voices.is_file() and paths.bible.is_file()):
        raise HTTPException(status_code=404, detail="cast not available yet; wait for the cast stage")
    cast = Cast.model_validate(read_json(paths.cast)) if paths.cast.is_file() else None
    bible = CastBible.model_validate(read_json(paths.bible))
    voice_ids = {voice.id for voice in _voices(paths)}
    for name, voice_id in body.overrides.items():
        if not _known_character(name, cast, bible):
            raise HTTPException(status_code=400, detail=f"unknown character {name!r}")
        if voice_id not in voice_ids:
            raise HTTPException(status_code=400, detail=f"unknown voice id {voice_id!r} for {name!r}")
    merged: dict[str, str] = {}
    if paths.cast_overrides.is_file():
        stored = read_json(paths.cast_overrides)
        if isinstance(stored, dict):
            merged.update({str(k): str(v) for k, v in stored.items()})
    merged.update(body.overrides)
    write_json(paths.cast_overrides, merged)
    # A queued job is already in the queue (retry / boot-time resubmit / waiting behind other jobs),
    # so it is not submitted again; but its cast stage row may already be done (retry from render
    # or finalize, orphan requeue), in which case the worker would skip stage_cast and never read
    # the overrides. Resetting the rows from cast onward makes the pending run honour the pin.
    _clear_render_outputs(paths)
    if job.status == JobStatus.queued:
        d.store.reset_stages_from(job_id, Stage.cast)
    else:
        d.store.requeue(job_id, from_stage=Stage.cast)
    d.store.append_event(job_id, "info", Stage.cast.value, f"re-cast requested: {', '.join(f'{k} -> {v}' for k, v in body.overrides.items())}")
    if job.status != JobStatus.queued:
        d.queue.submit(job_id)
    return JobAction(job_id=job_id, status=(d.store.get_job(job_id) or job).status.value)


# --------------------------------------------------------------------------- manifest, usage, artifacts
@router.get("/jobs/{job_id}/manifest", response_model=JobManifest)
def job_manifest(job_id: str, d: ApiDeps = Depends(deps)) -> JobManifest:
    """``manifest.json`` once finalized; before that a partial manifest of the chapters rendered so far."""
    job = _load_job(d.store, job_id)
    paths = _paths(d.settings, job_id)
    if paths.manifest.is_file():
        return JobManifest.model_validate(read_json(paths.manifest))
    if not paths.book.is_file():
        raise HTTPException(status_code=404, detail="no manifest yet")
    book = Book.model_validate(read_json(paths.book))
    chapters = [
        ChapterManifest.model_validate(read_json(paths.chapter_manifest(chapter.index)))
        for chapter in book.chapters
        if paths.chapter_manifest(chapter.index).is_file()
    ]
    if not chapters:
        raise HTTPException(status_code=404, detail="no chapter rendered yet")
    return build_job_manifest(job, book, chapters, d.providers.describe(), ["partial"])


@router.get("/jobs/{job_id}/usage", response_model=UsageOut)
def job_usage(job_id: str, d: ApiDeps = Depends(deps)) -> UsageOut:
    _load_job(d.store, job_id)
    summary = d.store.usage_summary(job_id)
    return UsageOut(
        calls=summary.calls, cache_hits=summary.cache_hits, cost_usd=summary.cost_usd,
        by_capability=summary.by_capability, events_count=d.store.usage_count(job_id),
    )


@router.get("/jobs/{job_id}/artifacts", response_model=ArtifactList)
def list_artifacts(job_id: str, request: Request, d: ApiDeps = Depends(deps)) -> ArtifactList:
    """Every file under the job directory except the uploaded source (and temp files)."""
    _load_job(d.store, job_id)
    paths = _paths(d.settings, job_id)
    files: list[ArtifactFile] = []
    if paths.job_dir.is_dir():
        for path in sorted(paths.job_dir.rglob("*")):
            if path.is_symlink() or not path.is_file() or _hidden_artifact(paths, path):
                continue
            rel = paths.relative(path)
            files.append(ArtifactFile(path=rel, bytes=path.stat().st_size, url=f"{_root(request)}/api/jobs/{job_id}/artifacts/{rel}"))
    return ArtifactList(files=files)


@router.get("/jobs/{job_id}/artifacts/{path:path}")
def get_artifact(job_id: str, path: str, d: ApiDeps = Depends(deps)) -> FileResponse:
    """Serve one artifact with Range support (so ``<audio>`` can seek); 400 if the path escapes the job dir."""
    _load_job(d.store, job_id)
    paths = _paths(d.settings, job_id)
    try:
        job_dir = paths.job_dir.resolve()
        target = (paths.job_dir / path).resolve()
    except (ValueError, OSError) as exc:                 # e.g. an embedded NUL byte from %00
        raise HTTPException(status_code=400, detail="invalid artifact path") from exc
    if target != job_dir and job_dir not in target.parents:
        raise HTTPException(status_code=400, detail="artifact path escapes the job directory")
    if not target.is_file() or _hidden_artifact(paths, target):
        raise HTTPException(status_code=404, detail=f"no artifact {path!r}")
    return FileResponse(target, media_type=_media_type(target))


# --------------------------------------------------------------------------- actions
@router.post("/jobs/{job_id}/retry", status_code=202, response_model=JobAction)
def retry_job(job_id: str, body: RetryRequest | None = None, d: ApiDeps = Depends(deps)) -> JobAction:
    """Requeue a failed or cancelled job (or any finished job from *from_stage* onward)."""
    job = _load_job(d.store, job_id)
    from_stage = body.from_stage if body else None
    if job.status in (JobStatus.running, JobStatus.queued):
        raise HTTPException(status_code=409, detail=f"job {job_id} is {job.status.value}")
    if job.status == JobStatus.done and from_stage is None:
        raise HTTPException(status_code=409, detail="job is done; pass from_stage to re-run part of it")
    if from_stage is not None:
        _paths(d.settings, job_id).manifest.unlink(missing_ok=True)
    d.store.requeue(job_id, from_stage=from_stage)
    d.store.append_event(job_id, "info", from_stage.value if from_stage else None, "retry requested" + (f" from stage {from_stage.value}" if from_stage else ""))
    d.queue.submit(job_id)
    return JobAction(job_id=job_id, status=(d.store.get_job(job_id) or job).status.value)


@router.post("/jobs/{job_id}/cancel", status_code=202, response_model=JobAction)
def cancel_job(job_id: str, d: ApiDeps = Depends(deps)) -> JobAction:
    """Queued jobs are cancelled at once; running jobs stop at their next checkpoint. 409 when terminal."""
    job = _load_job(d.store, job_id)
    if job.status in TERMINAL_STATUSES:
        raise HTTPException(status_code=409, detail=f"job {job_id} is already {job.status.value}")
    d.store.request_cancel(job_id)
    if job.status == JobStatus.queued:
        _discard_from_queue(d.queue, job_id)
        error = JobError(stage="queued", error_type="cancelled", message="cancelled while queued", retryable=True)
        d.store.set_status(job_id, JobStatus.cancelled, error=error)
        d.store.append_event(job_id, "warn", None, "cancelled while queued")
    return JobAction(job_id=job_id, status=(d.store.get_job(job_id) or job).status.value)


@router.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: str, d: ApiDeps = Depends(deps)) -> None:
    """Remove the rows and the job directory (the shared cache is untouched). 409 if running."""
    job = _load_job(d.store, job_id)
    _require_not_running(job)
    _discard_from_queue(d.queue, job_id)
    d.store.delete_job(job_id)
    shutil.rmtree(_paths(d.settings, job_id).job_dir, ignore_errors=True)
    log.info("job %s deleted", job_id)


# --------------------------------------------------------------------------- health
@router.get("/health", response_model=HealthOut)
def health(d: ApiDeps = Depends(deps)) -> HealthOut:
    """Provider checks, ffmpeg presence, queue depth and whether the data directory is writable."""
    providers = describe_providers(d.settings)
    writable = _writable(d.settings.data_dir)
    running_fn = getattr(d.queue, "running", None)
    queue = QueueOut(
        backend=str(getattr(d.queue, "backend", type(d.queue).__name__)),
        depth=d.queue.depth(),
        running=list(running_fn()) if callable(running_fn) else [],
    )
    return HealthOut(
        ok=all(entry.get("ok") for entry in providers) and writable,
        version=__version__,
        providers=providers,
        ffmpeg=ffmpeg_path() is not None,
        queue=queue,
        data_dir_writable=writable,
    )


def _writable(data_dir: Path) -> bool:
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        probe = tempfile.NamedTemporaryFile(dir=data_dir, prefix=".health-", delete=True)
        probe.write(b"ok")
        probe.close()
        return True
    except OSError:
        return False


__all__ = ["router", "deps", "ApiDeps", "overall_pct"]
