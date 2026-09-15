"""bookreader.jobs.paths - the on-disk layout of the data directory and one job's workspace.

This module is the only place file names are spelled. Everything under ``data/jobs/<id>/`` is
job-local; ``data/cache/`` is shared between jobs and ``data/bookreader.db`` holds job state.
Atomic writers (temp file in the same directory, then ``os.replace``) keep readers from ever
seeing a half-written file.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

DB_FILENAME = "bookreader.db"
CACHE_DIRNAME = "cache"
JOBS_DIRNAME = "jobs"
SOURCE_STEM = "source"
CHAPTER_DIR_FORMAT = "{index:02d}"
SCRIPT_FORMAT = "ch{index:02d}.json"
RENDER_KEY_FILENAME = "render.key"

_UMASK = os.umask(0)          # read once at import (os.umask is process-wide, not thread-safe)
os.umask(_UMASK)


def open_temp_beside(path: Path) -> tuple[int, Path]:
    """A uniquely named, exclusively created temp file next to *path* for an atomic replace.

    ``mkstemp`` gives every caller its own name, so concurrent writers of the same target (the
    worker threads of one process share a pid) never truncate or unlink each other's file: the
    last ``os.replace`` wins and readers only ever see complete files. The mode follows the
    process umask like an ordinary ``open`` would (mkstemp itself creates 0600 files).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        os.fchmod(fd, 0o666 & ~_UMASK)
    except OSError:
        pass
    return fd, Path(tmp_name)


def atomic_write_bytes(path: Path, data: bytes) -> Path:
    """Write *data* to *path* atomically; the parent directory is created when missing."""
    path = Path(path)
    fd, tmp = open_temp_beside(path)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def atomic_write_text(path: Path, text: str) -> Path:
    """Write *text* (UTF-8) to *path* atomically."""
    return atomic_write_bytes(path, text.encode("utf-8"))


def write_json(path: Path, obj: Any) -> Path:
    """Serialize a pydantic model or a plain JSON-able object to *path* atomically (indented)."""
    if isinstance(obj, BaseModel):
        text = obj.model_dump_json(indent=2)
    else:
        text = json.dumps(obj, indent=2, ensure_ascii=False)
    return atomic_write_text(path, text)


def read_json(path: Path) -> Any:
    """Load the JSON document at *path*."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


@dataclass(frozen=True)
class JobPaths:
    """Every file of one job, plus the shared locations of the data directory."""

    data_dir: Path
    job_id: str

    # ------------------------------------------------------------------ shared locations
    @property
    def db_path(self) -> Path:
        return self.data_dir / DB_FILENAME

    @property
    def cache_root(self) -> Path:
        return self.data_dir / CACHE_DIRNAME

    @property
    def jobs_root(self) -> Path:
        return self.data_dir / JOBS_DIRNAME

    # ------------------------------------------------------------------ job workspace
    @property
    def job_dir(self) -> Path:
        return self.jobs_root / self.job_id

    def source(self, suffix: str) -> Path:
        """``source<suffix>``; *suffix* includes the dot (``".txt"``)."""
        return self.job_dir / f"{SOURCE_STEM}{suffix}"

    def find_source(self) -> Path | None:
        """The uploaded file, whatever its extension, or None when nothing was uploaded."""
        if not self.job_dir.is_dir():
            return None
        matches = sorted(p for p in self.job_dir.glob(f"{SOURCE_STEM}.*") if p.is_file())
        return matches[0] if matches else None

    @property
    def book(self) -> Path:
        return self.job_dir / "book.json"

    @property
    def estimate(self) -> Path:
        return self.job_dir / "estimate.json"

    @property
    def bible(self) -> Path:
        return self.job_dir / "bible.json"

    @property
    def scripts_dir(self) -> Path:
        return self.job_dir / "scripts"

    def script(self, chapter_index: int) -> Path:
        return self.scripts_dir / SCRIPT_FORMAT.format(index=chapter_index)

    @property
    def voices(self) -> Path:
        return self.job_dir / "voices.json"

    @property
    def cast(self) -> Path:
        return self.job_dir / "cast.json"

    @property
    def cast_overrides(self) -> Path:
        return self.job_dir / "cast_overrides.json"

    @property
    def chapters_dir(self) -> Path:
        return self.job_dir / "chapters"

    def chapter_dir(self, chapter_index: int) -> Path:
        return self.chapters_dir / CHAPTER_DIR_FORMAT.format(index=chapter_index)

    def tts(self, chapter_index: int) -> Path:
        return self.chapter_dir(chapter_index) / "tts.json"

    def timeline(self, chapter_index: int) -> Path:
        return self.chapter_dir(chapter_index) / "timeline.json"

    def render_key(self, chapter_index: int) -> Path:
        """Sidecar holding the content key the chapter's outputs were rendered from."""
        return self.chapter_dir(chapter_index) / RENDER_KEY_FILENAME

    def mix(self, chapter_index: int) -> Path:
        return self.chapter_dir(chapter_index) / "mix.wav"

    def voice(self, chapter_index: int) -> Path:
        return self.chapter_dir(chapter_index) / "voice.wav"

    def music(self, chapter_index: int) -> Path:
        return self.chapter_dir(chapter_index) / "music.wav"

    def sfx(self, chapter_index: int) -> Path:
        return self.chapter_dir(chapter_index) / "sfx.wav"

    def mp3(self, chapter_index: int) -> Path:
        return self.chapter_dir(chapter_index) / "mix.mp3"

    def chapter_manifest(self, chapter_index: int) -> Path:
        return self.chapter_dir(chapter_index) / "manifest.json"

    def chapter_outputs(self, chapter_index: int) -> tuple[Path, ...]:
        """The files whose presence means the chapter is fully rendered."""
        return (
            self.mix(chapter_index),
            self.voice(chapter_index),
            self.music(chapter_index),
            self.sfx(chapter_index),
            self.chapter_manifest(chapter_index),
        )

    @property
    def manifest(self) -> Path:
        return self.job_dir / "manifest.json"

    @property
    def usage(self) -> Path:
        return self.job_dir / "usage.json"

    @property
    def log(self) -> Path:
        return self.job_dir / "job.log"

    def relative(self, path: Path) -> str:
        """*path* relative to the job directory as a POSIX string (the manifest's file references)."""
        return Path(path).relative_to(self.job_dir).as_posix()
