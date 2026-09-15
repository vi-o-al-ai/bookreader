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
from typing import AsyncIterator, Awaitable, Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response

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
MB = 1024 * 1024
UPLOAD_FORM_OVERHEAD_BYTES = 64 * 1024     # multipart framing plus the title/options fields
UPLOAD_PATH = "/api/jobs"

ASGIApp = Callable[[dict, Callable, Callable], Awaitable[None]]


class UploadLimitMiddleware:
    """Pure-ASGI guard for ``POST /api/jobs`` (streaming-safe, unlike ``BaseHTTPMiddleware``).

    FastAPI parses the multipart body into an ``UploadFile`` before the endpoint runs, so the
    endpoint's own byte check would only fire after the whole body was received and spooled to
    disk. Here a ``Content-Length`` above the limit is answered 413 before any body byte is
    read, and the ``receive`` callable is wrapped so a body that grows past the limit (chunked,
    or a lying header) raises ``HTTPException(413)`` from inside the form parser; FastAPI
    re-raises HTTP exceptions from body parsing untouched and the response goes out before the
    rest of the body is drained.
    """

    def __init__(self, app: ASGIApp, limit_bytes: int, path: str = UPLOAD_PATH) -> None:
        self.app = app
        self.limit_bytes = int(limit_bytes)
        self.path = path

    def _guarded(self, scope: dict) -> bool:
        if scope.get("type") != "http" or scope.get("method") != "POST":
            return False
        path = str(scope.get("path", ""))
        root = str(scope.get("root_path", "")).rstrip("/")
        if root and path.startswith(root):
            path = path[len(root):]
        return path.rstrip("/") == self.path

    def _detail(self) -> str:
        return f"upload exceeds {self.limit_bytes // MB} MB (BOOKREADER_MAX_UPLOAD_MB)"

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if not self._guarded(scope):
            await self.app(scope, receive, send)
            return
        allowed = self.limit_bytes + UPLOAD_FORM_OVERHEAD_BYTES
        declared = None
        for name, value in scope.get("headers", []):
            if name.lower() == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = None
                break
        if declared is not None and declared > allowed:
            response = JSONResponse({"detail": self._detail()}, status_code=413)
            await response(scope, receive, send)
            return
        received = 0

        async def limited_receive() -> dict:
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > allowed:
                    raise HTTPException(status_code=413, detail=self._detail())
            return message

        await self.app(scope, limited_receive, send)


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
    app.add_middleware(UploadLimitMiddleware, limit_bytes=settings.max_upload_mb * MB)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(INDEX_HTML, media_type="text/html; charset=utf-8")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:                     # browsers request it on every load; keep the console and access log clean
        return Response(status_code=204)

    return app
