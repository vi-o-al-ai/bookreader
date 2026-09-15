"""bookreader.api.app - ``create_app``: the FastAPI application and its lifespan.

Startup opens the SQLite job store, validates every selected provider (a
``ProviderConfigError`` is logged and re-raised so uvicorn exits non-zero), builds the process
singletons behind a :class:`bookreader.usage.UsageRouter` sink (so per-job ledgers created by
``run_job`` receive the usage rows), warms them up when configured, starts the queue and
resubmits jobs left queued or running by a previous process. Shutdown stops the queue with a
30 s grace period; a running job returns to ``queued`` and resumes on the next boot.
"""
from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Callable

from fastapi import FastAPI
from fastapi.responses import FileResponse

from bookreader import __version__
from bookreader.api.routes import router
from bookreader.jobs.db import JobStore
from bookreader.jobs.paths import JobPaths
from bookreader.jobs.queue import build_queue
from bookreader.pipeline.run import run_job
from bookreader.providers.base import build_providers, validate_providers, warmup_providers
from bookreader.settings import Settings
from bookreader.types import JobStatus, ProviderConfigError
from bookreader.usage import UsageRouter

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
INDEX_HTML = WEB_DIR / "index.html"
QUEUE_STOP_TIMEOUT_S = 30.0


def _resubmit_pending(store: JobStore, submit: Callable[[str], None]) -> list[str]:
    """Hand jobs left ``running`` (crash) or ``queued`` (clean shutdown) back to the queue."""
    ids: list[str] = list(store.requeue_orphans())
    for job in store.list_jobs(limit=100_000):
        if job.status == JobStatus.queued and job.id not in ids:
            ids.append(job.id)
    for job_id in ids:
        submit(job_id)
    if ids:
        log.info("resubmitted %d pending job(s): %s", len(ids), ", ".join(ids))
    return ids


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application; *settings* defaults to ``Settings.from_env()`` (the uvicorn factory path)."""
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
        store = JobStore(JobPaths(settings.data_dir, "").db_path)
        try:
            for report in validate_providers(settings):
                for warning in report.warnings:
                    log.warning("%s provider '%s': %s", report.capability, report.family, warning)
        except ProviderConfigError as exc:
            log.error("%s", exc)
            raise
        providers = build_providers(settings, UsageRouter())
        if settings.warmup:
            warmup_providers(providers)
        log.info("providers: %s", ", ".join(f"{cap}={desc['family']}" for cap, desc in providers.describe().items()))

        def run(job_id: str, stop: threading.Event) -> None:
            run_job(job_id, settings, store, providers, stop)

        queue = build_queue(settings, run)
        app.state.settings = settings
        app.state.store = store
        app.state.providers = providers
        app.state.queue = queue
        queue.start()
        _resubmit_pending(store, queue.submit)
        try:
            yield
        finally:
            queue.stop(QUEUE_STOP_TIMEOUT_S)
            store.close()

    app = FastAPI(title="bookreader", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.include_router(router)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(INDEX_HTML, media_type="text/html; charset=utf-8")

    return app
