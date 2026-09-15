"""bookreader.cli - the ``bookreader`` command: run, estimate, providers, serve, status.

``run`` drives one book through the pipeline in-process (inline worker) and prints the manifest
path; ``estimate`` ingests and chunks without touching any provider; ``providers`` prints the
startup checks; ``serve`` starts uvicorn on the API factory; ``status`` inspects a job in a data
directory. Exit codes: 0 ok, 1 the job failed, 2 bad input or configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import threading
from pathlib import Path
from typing import Any, Sequence, TextIO

from bookreader import __version__
from bookreader.analysis.chunker import make_chunks
from bookreader.ingest import load_book
from bookreader.jobs.db import JobStore
from bookreader.jobs.paths import JobPaths
from bookreader.pipeline.run import create_job, run_job
from bookreader.providers.base import build_providers, describe_providers, validate_providers, warmup_providers
from bookreader.settings import Settings
from bookreader.types import STAGE_ORDER, Estimate, InputError, Job, JobOptions, JobStatus, ProviderConfigError, Stage, StageState
from bookreader.usage import UsageLedger, UsageRouter

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
PROGRESS_POLL_S = 0.25
ANALYSIS_TOKENS_PER_CHUNK = 900     # same pre-flight assumptions as the ingest stage
CHARS_PER_TOKEN = 4
PROVIDER_FLAGS: dict[str, str] = {"analysis": "analysis_provider", "tts": "tts_provider", "music": "music_provider", "sfx": "sfx_provider"}


# --------------------------------------------------------------------------- helpers
def parse_chapters(text: str | None) -> list[int] | None:
    """``"1-3,5"`` -> ``[1, 2, 3, 5]``; None or empty -> None (all chapters)."""
    if not text or not text.strip():
        return None
    chapters: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        low, _, high = part.partition("-")
        try:
            start = int(low)
            end = int(high) if high else start
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid chapter range {part!r}; use forms like 3 or 1-4") from exc
        if start < 1 or end < start:
            raise argparse.ArgumentTypeError(f"invalid chapter range {part!r}")
        chapters.update(range(start, end + 1))
    return sorted(chapters) or None


def _settings_from_args(args: argparse.Namespace, **extra: Any) -> Settings:
    """Process settings from the environment with the CLI flags layered on top."""
    overrides: dict[str, Any] = dict(extra)
    for flag, field_name in PROVIDER_FLAGS.items():
        value = getattr(args, flag, None)
        if value:
            overrides[field_name] = value
    return Settings.from_env().with_overrides(**overrides)


def _configure_logging(settings: Settings) -> None:
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO), format="%(levelname)s %(name)s: %(message)s")


def _print_table(rows: list[list[str]], out: TextIO) -> None:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        out.write("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip() + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ProgressPrinter:
    """Background thread that prints a line whenever a stage record changes while ``run_job`` works.

    Polling every 250 ms matches the store's own progress throttle; a stage that starts and
    finishes between two polls still shows up as its ``done`` transition.
    """

    def __init__(self, store: JobStore, job_id: str, out: TextIO) -> None:
        self.store = store
        self.job_id = job_id
        self.out = out
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="bookreader-progress", daemon=True)
        self._seen: dict[str, tuple[str, int, int, str]] = {}

    def __enter__(self) -> "ProgressPrinter":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self._tick()

    def _tick(self) -> None:
        for record in self.store.stage_records(self.job_id):
            if record.state == StageState.pending:
                continue
            current = (record.state.value, record.done, record.total, record.message)
            if self._seen.get(record.stage.value) == current:
                continue
            self._seen[record.stage.value] = current
            progress = f" {record.done}/{record.total}" if record.total else ""
            suffix = f" {record.message}" if record.message else ""
            self.out.write(f"[{record.stage.value}] {record.state.value}{progress}{suffix}\n")
            self.out.flush()

    def _loop(self) -> None:
        while not self._stop.wait(PROGRESS_POLL_S):
            self._tick()


# --------------------------------------------------------------------------- commands
def cmd_run(args: argparse.Namespace) -> int:
    """Run one book end to end and print the manifest path."""
    source = Path(args.file)
    if not source.is_file():
        print(f"InputError: cannot read {source}: no such file", file=sys.stderr)
        return EXIT_USAGE
    settings = _settings_from_args(args, data_dir=Path(args.out), worker_mode="inline")
    _configure_logging(settings)
    try:
        validate_providers(settings)
    except ProviderConfigError as exc:
        print(f"ProviderConfigError: {exc}", file=sys.stderr)
        return EXIT_USAGE
    providers = build_providers(settings, UsageRouter())
    if settings.warmup:
        warmup_providers(providers)

    overrides: dict[str, str] = {}
    if args.cast:
        try:
            loaded = json.loads(Path(args.cast).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"InputError: cannot read cast overrides {args.cast}: {exc}", file=sys.stderr)
            return EXIT_USAGE
        if not isinstance(loaded, dict):
            print("InputError: cast overrides must be a JSON object of character -> voice id", file=sys.stderr)
            return EXIT_USAGE
        overrides = {str(k): str(v) for k, v in loaded.items()}
    options = JobOptions(chapters=args.chapters, music=not args.no_music, sfx=not args.no_sfx, cast_overrides=overrides)

    store = JobStore(JobPaths(settings.data_dir, "").db_path)
    from_stage = Stage(args.from_stage) if args.from_stage else None
    if from_stage is not None:
        job = _previous_job(store, source)
        if job is None:
            print(f"InputError: no previous job for {source.name} under {settings.data_dir}; run without --from-stage first", file=sys.stderr)
            return EXIT_USAGE
        store.requeue(job.id, from_stage=from_stage)
        JobPaths(settings.data_dir, job.id).manifest.unlink(missing_ok=True)
        print(f"job {job.id}: re-running from stage {from_stage.value}")
    else:
        job = create_job(store, settings, source, title=args.title, options=options)
        print(f"job {job.id}: {source.name} -> {settings.data_dir}")

    with ProgressPrinter(store, job.id, sys.stdout):
        job = run_job(job.id, settings, store, providers)
    return _report(job, JobPaths(settings.data_dir, job.id), store)


def _previous_job(store: JobStore, source: Path) -> Job | None:
    """The newest job in *store* created from the same file contents (for ``--from-stage``)."""
    digest = _sha256(source)
    for job in store.list_jobs():
        if job.source_sha256 == digest:
            return job
    return None


def _report(job: Job, paths: JobPaths, store: JobStore) -> int:
    if job.status == JobStatus.done:
        summary = store.usage_summary(job.id)
        print(f"done: {paths.manifest} ({summary.calls} provider calls, {summary.cache_hits} cache hits, ${summary.cost_usd:.4f})")
        return EXIT_OK
    error = job.error
    if error is None:
        print(f"job {job.id} ended with status {job.status.value}", file=sys.stderr)
        return EXIT_FAILED
    unit = f" (unit {error.unit})" if error.unit else ""
    retry = "retryable" if error.retryable else "not retryable"
    print(f"job {job.id} {job.status.value} at stage {error.stage}{unit}: {error.error_type}: {error.message} [{retry}]", file=sys.stderr)
    return EXIT_FAILED


def cmd_estimate(args: argparse.Namespace) -> int:
    """Ingest and chunk the book, then print the pre-flight counts and cost estimate."""
    source = Path(args.file)
    settings = _settings_from_args(args)
    try:
        book = load_book(source, title_hint=args.title)
    except InputError as exc:
        print(f"InputError: {exc}", file=sys.stderr)
        return EXIT_USAGE
    chapters = parse_chapters(args.chapters)
    if chapters:
        book.chapters = [chapter for chapter in book.chapters if chapter.index in set(chapters)]
    chunks = sum(len(make_chunks(chapter, settings.analysis_chunk_chars)) for chapter in book.chapters)
    chars = sum(len(paragraph.text) for chapter in book.chapters for paragraph in chapter.paragraphs)
    estimate = Estimate(
        chapters=len(book.chapters),
        paragraphs=sum(len(chapter.paragraphs) for chapter in book.chapters),
        words=book.word_count,
        chars=chars,
        quote_spans=sum(1 for chapter in book.chapters for span in chapter.spans if span.kind == "quote"),
        chunks=chunks,
        tts_chars=sum(len(span.text) for chapter in book.chapters for span in chapter.spans),
        analysis_input_tokens_est=chars // CHARS_PER_TOKEN + ANALYSIS_TOKENS_PER_CHUNK * chunks,
    )
    # UsageLedger.estimate_cost only reads the price table; no store is needed for a pre-flight estimate.
    estimate.cost_usd = UsageLedger(None, "", settings.prices).estimate_cost(estimate, settings.provider_families())  # type: ignore[arg-type]
    print(f"{book.title} ({source.name}, quote style {book.quote_style})")
    rows = [
        ["chapters", str(estimate.chapters)],
        ["paragraphs", str(estimate.paragraphs)],
        ["words", str(estimate.words)],
        ["chars", str(estimate.chars)],
        ["quote spans", str(estimate.quote_spans)],
        ["chunks", str(estimate.chunks)],
        ["tts chars", str(estimate.tts_chars)],
        ["analysis input tokens (est.)", str(estimate.analysis_input_tokens_est)],
    ]
    _print_table(rows, sys.stdout)
    families = settings.provider_families()
    print("estimated cost (USD):")
    _print_table([[f"  {cap}", families[cap], f"{cost:.4f}"] for cap, cost in estimate.cost_usd.items()], sys.stdout)
    print(f"  total  {sum(estimate.cost_usd.values()):.4f}")
    return EXIT_OK


def cmd_providers(args: argparse.Namespace) -> int:
    """Print the provider checks; exit 2 when any selected provider is not usable."""
    settings = _settings_from_args(args)
    rows = [["capability", "family", "class", "status", "notes"]]
    for entry in describe_providers(settings):
        notes = entry["error"] or "; ".join(entry["warnings"])
        rows.append([entry["capability"], entry["family"], entry.get("class", "-"), "ok" if entry["ok"] else "ERROR", notes or ""])
    _print_table(rows, sys.stdout)
    try:
        validate_providers(settings)
    except ProviderConfigError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return EXIT_USAGE
    return EXIT_OK


def cmd_serve(args: argparse.Namespace) -> int:
    """Validate the configuration, then run uvicorn on the app factory."""
    settings = Settings.from_env()
    _configure_logging(settings)
    try:
        validate_providers(settings)
    except ProviderConfigError as exc:
        print(f"ProviderConfigError: {exc}", file=sys.stderr)
        return EXIT_USAGE
    import uvicorn

    uvicorn.run("bookreader.api.app:create_app", factory=True, host=args.host, port=args.port, reload=args.reload, log_level=settings.log_level.lower())
    return EXIT_OK


def cmd_status(args: argparse.Namespace) -> int:
    """Print one job's status, stages and error from the data directory."""
    data_dir = Path(args.data_dir) if args.data_dir else Settings.from_env().data_dir
    paths = JobPaths(data_dir, args.job_id)
    if not paths.db_path.is_file():
        print(f"no database at {paths.db_path}", file=sys.stderr)
        return EXIT_USAGE
    store = JobStore(paths.db_path)
    job = store.get_job(args.job_id)
    if job is None:
        print(f"unknown job {args.job_id!r} in {data_dir}", file=sys.stderr)
        return EXIT_FAILED
    print(f"{job.id}  {job.title}  ({job.filename})")
    print(f"status: {job.status.value}  stage: {job.stage.value if job.stage else '-'}  created: {job.created_at}  finished: {job.finished_at or '-'}")
    rows = [["stage", "state", "progress", "attempts", "message"]]
    for record in store.stage_records(job.id):
        rows.append([record.stage.value, record.state.value, f"{record.done}/{record.total}", str(record.attempts), record.message])
    _print_table(rows, sys.stdout)
    if job.error:
        print(f"error: {job.error.error_type} at {job.error.stage}: {job.error.message}" + (f" (unit {job.error.unit})" if job.error.unit else ""))
    summary = store.usage_summary(job.id)
    print(f"usage: {summary.calls} calls, {summary.cache_hits} cache hits, ${summary.cost_usd:.4f}")
    if paths.manifest.is_file():
        print(f"manifest: {paths.manifest}")
    return EXIT_OK


# --------------------------------------------------------------------------- parser
def _add_provider_flags(parser: argparse.ArgumentParser) -> None:
    for flag in PROVIDER_FLAGS:
        parser.add_argument(f"--{flag}", metavar="FAMILY", help=f"{flag} provider family (mock|anthropic|elevenlabs|local)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bookreader", description="Turn a book into an audio drama with character voices, music and sound effects.")
    parser.add_argument("--version", action="version", version=f"bookreader {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="render one book in-process and print the manifest path")
    run.add_argument("file", help="book file (.txt .md .epub .pdf)")
    run.add_argument("--out", default="./data", metavar="DIR", help="data directory (default ./data)")
    run.add_argument("--title", help="book title (default: detected from the text)")
    run.add_argument("--chapters", type=parse_chapters, help="chapter subset, e.g. 1-2 or 1,3")
    run.add_argument("--no-music", action="store_true", help="skip music beds")
    run.add_argument("--no-sfx", action="store_true", help="skip sound effects")
    run.add_argument("--cast", metavar="FILE", help="JSON object of character -> voice id overrides")
    run.add_argument("--from-stage", choices=[stage.value for stage in STAGE_ORDER], help="re-run the previous job for this file from a stage")
    _add_provider_flags(run)
    run.set_defaults(func=cmd_run)

    estimate = sub.add_parser("estimate", help="ingest and chunk without calling any provider; print counts and cost")
    estimate.add_argument("file")
    estimate.add_argument("--title")
    estimate.add_argument("--chapters", help="chapter subset, e.g. 1-2")
    _add_provider_flags(estimate)
    estimate.set_defaults(func=cmd_estimate)

    providers = sub.add_parser("providers", help="check the selected provider families")
    _add_provider_flags(providers)
    providers.set_defaults(func=cmd_providers)

    serve = sub.add_parser("serve", help="start the HTTP API and web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    status = sub.add_parser("status", help="show a job's stages and error")
    status.add_argument("job_id")
    status.add_argument("--data-dir", metavar="DIR", help="data directory (default BOOKREADER_DATA_DIR or ./data)")
    status.set_defaults(func=cmd_status)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``bookreader`` console script; returns the exit code."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:                     # argparse already printed the message
        return int(exc.code or 0)
    try:
        return int(args.func(args))
    except ValueError as exc:                     # bad BOOKREADER_* value or CLI override
        print(f"ConfigError: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover - console script entry
    sys.exit(main())
