"""bookreader.pipeline.cache - the content-addressed clip cache shared by every job.

Layout under the cache root: ``<kind>/<key>.wav`` for audio (tts, music, sfx) and
``<kind>/<key>.json`` for JSON documents (analysis results, voice catalogs). Keys come from
:func:`bookreader.types.clip_key` / :func:`bookreader.types.content_key`, so the same request
always maps to the same file. Reads (``get``, and ``has`` when it answers True) touch the file's
mtime, which makes :meth:`ClipCache.prune` an LRU eviction rather than a FIFO one; prune can also
be told to spare every entry newer than a point in time, so a job pruning at finalize never
evicts what another running job is relying on.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from bookreader.audio.pcm import read_wav, write_wav
from bookreader.jobs.paths import atomic_write_text
from bookreader.types import AudioClip, InputError

log = logging.getLogger(__name__)

AUDIO_SUFFIX = ".wav"
JSON_SUFFIX = ".json"


class ClipCache:
    """Content-addressed store for rendered clips and JSON results under *root*."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # ------------------------------------------------------------------ paths
    def path(self, kind: str, key: str, suffix: str = AUDIO_SUFFIX) -> Path:
        """Location of the entry *key* of *kind*; the directory is created on demand by writers."""
        return self.root / kind / f"{key}{suffix}"

    def json_path(self, kind: str, key: str) -> Path:
        return self.path(kind, key, JSON_SUFFIX)

    # ------------------------------------------------------------------ audio
    def has(self, kind: str, key: str) -> bool:
        """Whether the clip is cached; a hit counts as a use (mtime bumped) like :meth:`get`."""
        path = self.path(kind, key)
        if not path.is_file():
            return False
        _touch(path)
        return True

    def get(self, kind: str, key: str) -> AudioClip | None:
        """The cached clip, or None when absent or unreadable (a corrupt file is dropped)."""
        path = self.path(kind, key)
        if not path.is_file():
            return None
        try:
            clip = read_wav(path)
        except InputError as exc:
            log.warning("dropping corrupt cache entry %s: %s", path, exc)
            path.unlink(missing_ok=True)
            return None
        _touch(path)
        return clip

    def put(self, kind: str, key: str, clip: AudioClip) -> Path:
        """Store *clip* atomically and return its path."""
        return write_wav(self.path(kind, key), clip)

    # ------------------------------------------------------------------ json
    def get_json(self, kind: str, key: str, max_age_s: float | None = None) -> Any | None:
        """The cached JSON document, or None when absent, older than *max_age_s* or corrupt."""
        path = self.json_path(kind, key)
        if not path.is_file():
            return None
        if max_age_s is not None and time.time() - path.stat().st_mtime > max_age_s:
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                value = json.load(fh)
        except (OSError, ValueError) as exc:
            log.warning("dropping corrupt cache entry %s: %s", path, exc)
            path.unlink(missing_ok=True)
            return None
        if max_age_s is None:            # a TTL entry must keep its write time, or it never expires
            _touch(path)
        return value

    def put_json(self, kind: str, key: str, obj: Any) -> Path:
        """Store a JSON-able *obj* (or a pydantic model) atomically and return its path."""
        text = obj.model_dump_json() if hasattr(obj, "model_dump_json") else json.dumps(obj, ensure_ascii=False)
        return atomic_write_text(self.json_path(kind, key), text)

    # ------------------------------------------------------------------ housekeeping
    def _files(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        return [p for p in self.root.rglob("*") if p.is_file() and not p.name.startswith(".")]

    def size(self) -> int:
        """Total bytes of every cached file."""
        return sum(p.stat().st_size for p in self._files())

    def prune(self, max_bytes: int, keep_newer_than: float | None = None) -> list[Path]:
        """Delete least-recently-used files until the cache fits *max_bytes* (0 = unbounded).

        Files used (mtime) at or after *keep_newer_than* (a ``time.time()`` stamp, typically the
        start of the oldest job still running) are never removed, so the cap may be exceeded
        temporarily rather than pulling clips out from under a concurrent job. Returns the paths
        removed, oldest first.
        """
        if max_bytes <= 0:
            return []
        entries = []
        for path in self._files():
            stat = path.stat()
            entries.append((stat.st_mtime, stat.st_size, path))
        total = sum(size for _, size, _ in entries)
        removed: list[Path] = []
        for mtime, size, path in sorted(entries, key=lambda item: (item[0], str(item[2]))):
            if total <= max_bytes:
                break
            if keep_newer_than is not None and mtime >= keep_newer_than:
                continue
            try:
                path.unlink()
            except OSError as exc:
                log.warning("cache prune could not remove %s: %s", path, exc)
                continue
            total -= size
            removed.append(path)
        if removed:
            log.info("cache pruned %d file(s); %d bytes remain under %s", len(removed), total, self.root)
        return removed


def _touch(path: Path) -> None:
    """Bump the mtime so LRU pruning sees the entry as recently used; failures are ignored."""
    try:
        os.utime(path, None)
    except OSError:
        pass
