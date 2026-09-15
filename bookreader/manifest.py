"""bookreader.manifest - assemble the job-level ``manifest.json`` (the public timing contract)."""
from __future__ import annotations

from typing import Mapping, Sequence

from bookreader.types import Book, ChapterManifest, Job, JobManifest


def build_job_manifest(
    job: Job,
    book: Book,
    chapter_manifests: Sequence[ChapterManifest],
    providers_desc: Mapping[str, Mapping[str, str]],
    warnings: Sequence[str],
) -> JobManifest:
    """Combine the rendered chapter manifests into one :class:`JobManifest`.

    Chapters are ordered by index; the total duration is their sum; the title is the book's
    detected title (falling back to the job title) and *providers_desc* is
    ``Providers.describe()``. Pass ``["partial"]`` in *warnings* for an in-progress manifest.
    """
    chapters = sorted(chapter_manifests, key=lambda chapter: chapter.index)
    return JobManifest(
        job_id=job.id,
        title=book.title or job.title,
        created_at=job.created_at,
        providers={capability: dict(desc) for capability, desc in providers_desc.items()},
        total_duration_ms=sum(chapter.duration_ms for chapter in chapters),
        chapters=chapters,
        warnings=list(warnings),
    )
