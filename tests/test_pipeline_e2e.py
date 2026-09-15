"""End-to-end pipeline test: one session-scoped run of the fixture book on the all-mock family."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from bookreader.audio.pcm import read_wav
from bookreader.jobs.db import JobStore
from bookreader.jobs.paths import JobPaths
from bookreader.pipeline.cache import ClipCache
from bookreader.pipeline.run import create_job, run_job
from bookreader.settings import Settings
from bookreader.types import (
    NARRATOR,
    STAGE_ORDER,
    Book,
    Cast,
    CastBible,
    ChapterManifest,
    Estimate,
    Job,
    JobManifest,
    JobStatus,
    StageState,
    UsageSummary,
)

EXPECTED_CHARACTERS = {"Mara Quill", "Tobias", "Ansel Vey", "Hetta"}


@dataclass
class Run:
    settings: Settings
    store: JobStore
    job: Job
    paths: JobPaths

    @property
    def manifest(self) -> JobManifest:
        return JobManifest.model_validate_json(self.paths.manifest.read_text(encoding="utf-8"))

    @property
    def book(self) -> Book:
        return Book.model_validate_json(self.paths.book.read_text(encoding="utf-8"))

    @property
    def cache(self) -> ClipCache:
        return ClipCache(self.paths.cache_root)


@pytest.fixture(scope="session")
def run(tmp_path_factory: pytest.TempPathFactory, sample_book_path: Path) -> Run:
    data_dir = tmp_path_factory.mktemp("e2e") / "data"
    settings = Settings.from_env({}).with_overrides(data_dir=data_dir, worker_mode="inline", mock_ms_per_char=4, warmup=False)
    store = JobStore(data_dir / "bookreader.db")
    job = create_job(store, settings, sample_book_path, title="The Lighthouse at Gull Point")
    job = run_job(job.id, settings, store)
    return Run(settings=settings, store=store, job=job, paths=JobPaths(data_dir, job.id))


def test_job_finished_with_every_stage_done(run: Run) -> None:
    assert run.job.status == JobStatus.done, run.job.error
    assert run.job.error is None and run.job.started_at and run.job.finished_at
    assert run.job.stage is not None and run.job.stage.value == "finalize"
    records = run.store.stage_records(run.job.id)
    assert [r.stage for r in records] == list(STAGE_ORDER)
    assert all(r.state == StageState.done and r.attempts == 1 and r.done == r.total > 0 for r in records)


def test_manifest_has_three_chapters_with_equal_length_stems(run: Run) -> None:
    manifest = run.manifest
    assert manifest.job_id == run.job.id and manifest.title == "The Lighthouse at Gull Point"
    assert [c.index for c in manifest.chapters] == [1, 2, 3]
    assert manifest.total_duration_ms == sum(c.duration_ms for c in manifest.chapters) > 0
    assert manifest.providers["tts"] == {"family": "mock", "cache_version": "1:4"}   # mock tts keys its pacing knob
    assert manifest.cast_file == "cast.json" and manifest.usage_file == "usage.json"
    for chapter in manifest.chapters:
        assert chapter.title.startswith(f"Chapter {chapter.index}")
        lengths = set()
        for name in ("mix", "voice", "music", "sfx"):
            rel = chapter.files[name]
            assert rel == f"chapters/{chapter.index:02d}/{name}.wav"
            clip = read_wav(run.paths.job_dir / rel)
            assert clip.sample_rate == 22050
            lengths.add(len(clip.samples))
        assert len(lengths) == 1 and lengths.pop() > 0
        assert "mp3" in chapter.files
        sidecar = ChapterManifest.model_validate_json(run.paths.chapter_manifest(chapter.index).read_text(encoding="utf-8"))
        assert sidecar.model_dump() == chapter.model_dump()


def test_segments_reproduce_every_span_in_order(run: Run) -> None:
    book = run.book
    for chapter, manifest in zip(book.chapters, run.manifest.chapters):
        assert [s.text for s in manifest.segments] == [span.text for span in chapter.spans]
        assert "".join(s.text for s in manifest.segments) == "".join(span.text for span in chapter.spans)
        assert [s.id for s in manifest.segments] == [span.id for span in chapter.spans]
        for segment in manifest.segments:
            assert segment.kind == ("narration" if segment.speaker == NARRATOR else "dialogue")
    speakers = {s.speaker for c in run.manifest.chapters for s in c.segments}
    assert speakers == EXPECTED_CHARACTERS | {NARRATOR}


def test_timings_monotonic_and_within_duration(run: Run) -> None:
    for chapter in run.manifest.chapters:
        previous_end = 0
        for segment in chapter.segments:
            assert 0 <= segment.start_ms <= segment.end_ms <= chapter.duration_ms
            assert segment.start_ms >= previous_end
            assert segment.end_ms > segment.start_ms or not segment.text.strip()
            previous_end = segment.end_ms
        for cue in chapter.cues:
            assert 0 <= cue.start_ms < cue.end_ms <= chapter.duration_ms


def test_every_cue_has_a_cached_clip(run: Run) -> None:
    cache = run.cache
    sfx_cues = music_cues = 0
    for chapter in run.manifest.chapters:
        for cue in chapter.cues:
            assert cache.has(cue.kind, cue.clip), f"{cue.id} ({cue.kind}) has no clip {cue.clip}"
            if cue.kind == "sfx":
                sfx_cues += 1
                assert cue.description and cue.sfx_kind in ("impact", "ambient") and cue.anchor_segment
            else:
                music_cues += 1
                assert cue.mood and cue.prompt
    assert sfx_cues > 0 and music_cues > 0
    for segment in run.manifest.chapters[0].segments:
        assert segment.voice_id.startswith("mock-")


def test_cast_and_bible(run: Run) -> None:
    cast = Cast.model_validate_json(run.paths.cast.read_text(encoding="utf-8"))
    assert cast.narrator.character == NARRATOR
    assert {a.character for a in cast.characters} == EXPECTED_CHARACTERS
    ids = [cast.narrator.voice.id] + [a.voice.id for a in cast.characters]
    assert len(set(ids)) == len(ids)
    assert cast.assignment_for("Tobias").voice.age == "child"
    bible = CastBible.model_validate_json(run.paths.bible.read_text(encoding="utf-8"))
    ansel = bible.find("Ansel Vey")
    assert ansel is not None and "the stranger" in ansel.aliases
    assert bible.find("the stranger") is ansel
    assert run.paths.voices.is_file() and len(json.loads(run.paths.voices.read_text(encoding="utf-8"))) == 16


def test_estimate_usage_and_log(run: Run) -> None:
    estimate = Estimate.model_validate_json(run.paths.estimate.read_text(encoding="utf-8"))
    assert estimate.chapters == 3 and estimate.chunks == 3 and estimate.quote_spans > 0
    assert estimate.tts_chars == sum(len(s.text) for c in run.manifest.chapters for s in c.segments)
    assert set(estimate.cost_usd) == {"analysis", "tts", "music", "sfx"}
    usage = UsageSummary.model_validate_json(run.paths.usage.read_text(encoding="utf-8"))
    assert usage.cache_hits == 0
    assert usage.calls > 0 and usage.cost_usd == 0.0
    assert usage.by_capability["tts"]["characters"] == estimate.tts_chars
    assert usage.model_dump() == run.store.usage_summary(run.job.id).model_dump()
    log_text = run.paths.log.read_text(encoding="utf-8")
    assert log_text.strip()
    assert "stage render done" in log_text and run.job.id in log_text


def test_chapter_one_mixed_before_chapter_three_renders(run: Run) -> None:
    events = run.store.events_after(run.job.id, 0, limit=1000)
    messages = [e.message for e in events]
    mixed_first = next(i for i, m in enumerate(messages) if m.startswith("chapter 1/3 mixed"))
    tts_third = next(i for i, m in enumerate(messages) if m.startswith("render/tts: ch 3/3"))
    assert mixed_first < tts_third
    assert [e.stage for e in events if e.message.startswith("stage ") and e.message.endswith(" started")] == [s.value for s in STAGE_ORDER]
    assert all(e.level == "info" for e in events)


def test_run_job_is_a_no_op_for_a_job_that_is_not_queued(run: Run) -> None:
    """A stale queue entry (cancelled, deleted+retried, or already running elsewhere) must not
    re-run a finished job: run_job claims the row atomically from ``queued`` only."""
    before = run.store.get_job(run.job.id)
    assert before is not None and before.status == JobStatus.done
    events_before = len(run.store.events_after(run.job.id, 0, limit=10_000))
    again = run_job(run.job.id, run.settings, run.store)
    assert again.status == JobStatus.done and again.finished_at == before.finished_at and again.started_at == before.started_at
    assert len(run.store.events_after(run.job.id, 0, limit=10_000)) == events_before
    assert all(r.attempts == 1 for r in run.store.stage_records(run.job.id))
    run.store.set_status(run.job.id, JobStatus.running)                      # "another worker holds it"
    assert run_job(run.job.id, run.settings, run.store).status == JobStatus.running
    assert len(run.store.events_after(run.job.id, 0, limit=10_000)) == events_before
    run.store.set_status(run.job.id, JobStatus.done)
