"""bookreader.pipeline.stages - the five stage functions: ingest, analyze, cast, render, finalize.

Every stage is idempotent and resumable: it skips work whose outputs already exist in the job
directory, and inside a stage every provider call goes through the content-addressed cache, so a
retried job only pays for what it has not produced yet. Stages raise typed errors with ``unit``
set to the failing work item (``ch02:c2p7s1``, ``ch01:chunk0``, ``ch03:c3m001``) and call
``ctx.check_cancelled()`` between units of work.
"""
from __future__ import annotations

import contextvars
import dataclasses
import logging
import re
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, TypeVar

from bookreader.analysis.assemble import assemble_script
from bookreader.analysis.bible import apply_updates, finalize, register_speaker
from bookreader.analysis.chunker import make_chunks
from bookreader.analysis.validate import validate_chunk_analysis
from bookreader.audio.dsp import normalize_rms, trim_silence
from bookreader.audio.export import export_mp3
from bookreader.audio.mixer import build_chapter_manifest, mix_chapter
from bookreader.audio.pcm import to_canonical, to_int16
from bookreader.audio.timeline import TimelineParams, build_timeline
from bookreader.casting import cast_voices, settings_for
from bookreader.ingest import load_book
from bookreader.jobs.paths import read_json, write_json
from bookreader.manifest import build_job_manifest
from bookreader.pipeline.context import JobContext
from bookreader.retry import FamilyLimiter, with_retry
from bookreader.types import (
    NARRATOR,
    SAMPLE_RATE,
    SCHEMA_VERSION,
    AudioClip,
    Book,
    BookreaderError,
    Cast,
    CastBible,
    ChapterManifest,
    ChapterScript,
    ChapterTimeline,
    ChapterTts,
    Chunk,
    ChunkAnalysis,
    Estimate,
    InputError,
    Mood,
    SfxJob,
    Stage,
    TTSRequest,
    TtsJob,
    VoiceInfo,
    clip_key,
    content_key,
)

log = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

VOICE_CATALOG_TTL_S = 24 * 3600
ANALYSIS_TOKENS_PER_CHUNK = 900              # prompt overhead per analyzer call (estimate only)
CHARS_PER_TOKEN = 4
NEIGHBOUR_TEXT_CHARS = 200                   # previous_text / next_text truncation
TTS_TRIM_THRESHOLD_DBFS = -45.0
TTS_TRIM_KEEP_MS = 40
TARGET_RMS_DBFS = -20.0
MB = 1024 * 1024
TRACK_KIND: dict[str, str] = {"voice": "tts", "music": "music", "sfx": "sfx"}
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class RenderError(BookreaderError):
    """A guard or consistency failure inside the render stage (reported as ``internal``)."""


# --------------------------------------------------------------------------- shared helpers
def _unit(chapter_index: int, item: str) -> str:
    return f"ch{chapter_index:02d}:{item}"


def _attach_unit(exc: BaseException, unit: str) -> None:
    """Tag *exc* with the failing work item unless a deeper layer already did."""
    if getattr(exc, "unit", None) is None:
        try:
            exc.unit = unit  # type: ignore[attr-defined]
        except AttributeError:
            pass


def _load_book(ctx: JobContext) -> Book:
    return Book.model_validate(read_json(ctx.paths.book))


def _load_script(ctx: JobContext, chapter_index: int) -> ChapterScript:
    return ChapterScript.model_validate(read_json(ctx.paths.script(chapter_index)))


def _load_cast(ctx: JobContext) -> Cast:
    return Cast.model_validate(read_json(ctx.paths.cast))


def _retry_logger(ctx: JobContext, unit: str) -> Callable[[int, BaseException, float], None]:
    def on_retry(attempt: int, exc: BaseException, delay: float) -> None:
        ctx.log("warn", f"{unit}: transient provider error (attempt {attempt}): {exc}; retrying in {delay:.1f}s")

    return on_retry


def _finish_clip(clip: AudioClip, trim: bool) -> AudioClip:
    """Canonical rate, optional silence trim, RMS normalization to the voice/bed target level."""
    clip = to_canonical(clip)
    samples = clip.samples
    if trim:
        samples = trim_silence(samples, TTS_TRIM_THRESHOLD_DBFS, TTS_TRIM_KEEP_MS, sample_rate=SAMPLE_RATE)
    samples = normalize_rms(samples, TARGET_RMS_DBFS, sample_rate=SAMPLE_RATE)
    return AudioClip(to_int16(samples), SAMPLE_RATE)


def _record_hit(ctx: JobContext, kind: str, key: str, family: str, unit_type: str, units: float, meta: dict[str, object]) -> None:
    """Record a cache-hit usage row unless this very run produced the entry (then its cost was
    already recorded by the provider call and reuse within the job is free by construction)."""
    if not ctx.produced(kind, key):
        ctx.ledger.record(kind, family, unit_type, units, cache_hit=True, meta=meta)


def _run_parallel(
    ctx: JobContext,
    items: list[T],
    work: Callable[[T], R],
    unit_of: Callable[[T], str],
    on_done: Callable[[T, R], None],
) -> None:
    """Run *work* over *items* on a thread pool of ``settings.concurrency`` workers.

    Cancellation is checked between submissions and after every completion; on cancel or error
    pending futures are dropped (``cancel_futures``) while in-flight calls finish and keep their
    cache entries. The first failure is re-raised tagged with the item's unit.
    """
    if not items:
        return
    pool = ThreadPoolExecutor(max_workers=max(1, ctx.settings.concurrency), initializer=ctx.register_thread)
    futures: dict[Future[R], T] = {}
    try:
        for item in items:
            ctx.check_cancelled()
            futures[pool.submit(contextvars.copy_context().run, work, item)] = item
        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
            except BaseException as exc:
                _attach_unit(exc, unit_of(item))
                raise
            on_done(item, result)
            ctx.check_cancelled()
    except BaseException:
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown(wait=True)


# --------------------------------------------------------------------------- ingest
def stage_ingest(ctx: JobContext) -> None:
    """Load the uploaded file into ``book.json`` and write the pre-flight ``estimate.json``."""
    paths = ctx.paths
    if paths.book.is_file():
        ctx.log("info", "ingest: book.json exists; skipped")
        ctx.progress("ingest", 1, 1, "book.json present")
        return
    source = paths.find_source()
    if source is None:
        raise InputError(f"no source file found in {paths.job_dir}")
    book = load_book(source, title_hint=ctx.job.title or None)

    wanted = ctx.job.options.chapters
    if wanted:
        keep = set(wanted)
        book.chapters = [chapter for chapter in book.chapters if chapter.index in keep]
        if not book.chapters:
            raise InputError(f"options.chapters {sorted(keep)} selects no chapter of the {len(keep)} requested")
    span_count = sum(len(chapter.spans) for chapter in book.chapters)
    if ctx.settings.max_segments and span_count > ctx.settings.max_segments:
        raise InputError(f"book has {span_count} spans; the limit is {ctx.settings.max_segments} (BOOKREADER_MAX_SEGMENTS)")

    chunk_count = sum(len(make_chunks(chapter, ctx.settings.analysis_chunk_chars)) for chapter in book.chapters)
    chars = sum(len(paragraph.text) for chapter in book.chapters for paragraph in chapter.paragraphs)
    tts_chars = sum(len(span.text) for chapter in book.chapters for span in chapter.spans)
    estimate = Estimate(
        chapters=len(book.chapters),
        paragraphs=sum(len(chapter.paragraphs) for chapter in book.chapters),
        words=book.word_count,
        chars=chars,
        quote_spans=sum(1 for chapter in book.chapters for span in chapter.spans if span.kind == "quote"),
        chunks=chunk_count,
        tts_chars=tts_chars,
        analysis_input_tokens_est=chars // CHARS_PER_TOKEN + ANALYSIS_TOKENS_PER_CHUNK * chunk_count,
    )
    estimate.cost_usd = ctx.ledger.estimate_cost(estimate, ctx.settings.provider_families())

    write_json(paths.book, book)
    write_json(paths.estimate, estimate)
    updates: dict[str, str] = {}
    if not ctx.job.source_sha256:
        updates["source_sha256"] = book.source_sha256
    if not ctx.job.title:
        updates["title"] = book.title
    if updates:
        ctx.store.update_job(ctx.job.id, **updates)
        ctx.job = ctx.job.model_copy(update=updates)
    ctx.log("info", f"ingest: {book.title!r}: {len(book.chapters)} chapter(s), {estimate.words} words, {span_count} spans, {chunk_count} chunk(s)")
    ctx.progress("ingest", 1, 1, f"{len(book.chapters)} chapters")


# --------------------------------------------------------------------------- analyze
def _analysis_key(ctx: JobContext, chunk: Chunk, bible: CastBible) -> str:
    analyzer = ctx.providers.analysis
    return content_key(
        "analysis", analyzer.family, str(analyzer.cache_version), str(analyzer.model_id),
        SCHEMA_VERSION, chunk.model_dump(mode="json"), bible.fingerprint(),
    )


def _analyze_chunk(ctx: JobContext, chunk: Chunk, bible: CastBible) -> ChunkAnalysis:
    """Cached, validated analysis of one chunk."""
    analyzer = ctx.providers.analysis
    key = _analysis_key(ctx, chunk, bible)
    cached = ctx.cache.get_json("analysis", key)
    if cached is not None:
        ctx.ledger.record("analysis", analyzer.family, "calls", 1.0, cache_hit=True, meta={"chapter": chunk.chapter_index, "chunk": chunk.chunk_index})
        return ChunkAnalysis.model_validate(cached)
    with FamilyLimiter.acquire(analyzer.family, ctx.settings.concurrency):
        raw = analyzer.analyze_chunk(chunk, bible)
    analysis, warnings = validate_chunk_analysis(raw, chunk, bible)
    for warning in warnings:
        ctx.log("warn", f"{_unit(chunk.chapter_index, f'chunk{chunk.chunk_index}')}: {warning}")
    ctx.cache.put_json("analysis", key, analysis)
    return analysis


def _last_mood(analysis: ChunkAnalysis, current: Mood) -> Mood:
    """Mood in force after *analysis* (the last start/change/stop action wins)."""
    for cue in analysis.music_cues:
        current = "none" if cue.action == "stop" else cue.mood
    return current


def stage_analyze(ctx: JobContext) -> None:
    """Label every chunk in book order, thread the cast bible through, assemble chapter scripts."""
    paths = ctx.paths
    book = _load_book(ctx)
    if paths.bible.is_file() and all(paths.script(chapter.index).is_file() for chapter in book.chapters):
        ctx.log("info", "analyze: bible.json and every script exist; skipped")
        ctx.progress("analyze", 1, 1, "scripts present")
        return

    chunks_by_chapter = {chapter.index: make_chunks(chapter, ctx.settings.analysis_chunk_chars) for chapter in book.chapters}
    total = sum(len(chunks) for chunks in chunks_by_chapter.values())
    bible = CastBible()
    analyses: dict[int, list[ChunkAnalysis]] = defaultdict(list)
    chapter_moods: dict[int, Mood] = {}
    mood: Mood = "none"
    done = 0
    ctx.progress("analyze", 0, total, f"chunk 0/{total}")
    for chapter in book.chapters:
        chapter_moods[chapter.index] = mood
        for chunk in chunks_by_chapter[chapter.index]:
            ctx.check_cancelled()
            chunk = chunk.model_copy(update={"prior_mood": mood})
            unit = _unit(chapter.index, f"chunk{chunk.chunk_index}")
            try:
                analysis = _analyze_chunk(ctx, chunk, bible)
            except BaseException as exc:
                _attach_unit(exc, unit)
                raise
            bible = apply_updates(bible, analysis.characters, chapter.index)
            quote_ids = {span.id for span in chunk.spans if span.kind == "quote"}
            for label in analysis.labels:
                if label.span_id in quote_ids and label.speaker != NARRATOR:
                    bible, _ = register_speaker(bible, label.speaker, chapter.index)
            mood = _last_mood(analysis, mood)
            analyses[chapter.index].append(analysis)
            write_json(paths.bible, bible)
            done += 1
            ctx.progress("analyze", done, total, f"chunk {done}/{total} (ch {chapter.index})")

    bible = finalize(bible)
    write_json(paths.bible, bible)
    for chapter in book.chapters:
        script = assemble_script(chapter, analyses[chapter.index], bible, chapter_moods[chapter.index])
        write_json(paths.script(chapter.index), script)
        for warning in script.warnings:
            ctx.log("warn", f"ch{chapter.index:02d}: {warning}")
    ctx.log("info", f"analyze: {total} chunk(s) -> {len(book.chapters)} script(s); cast bible has {len(bible.characters)} character(s)")
    ctx.progress("analyze", total, total, f"chunk {total}/{total}")


# --------------------------------------------------------------------------- cast
def _voice_catalog(ctx: JobContext) -> list[VoiceInfo]:
    """The provider's voices, cached under ``data/cache/voices/`` for 24 h."""
    tts = ctx.providers.tts
    key = f"{tts.family}-{tts.cache_version}"
    cached = ctx.cache.get_json("voices", key, max_age_s=VOICE_CATALOG_TTL_S)
    if isinstance(cached, list) and cached:
        return [VoiceInfo.model_validate(item) for item in cached]
    with FamilyLimiter.acquire(tts.family, ctx.settings.concurrency):
        voices = with_retry(tts.list_voices, on_retry=_retry_logger(ctx, "cast:list_voices"))[: ctx.settings.max_voices]
    if not voices:
        raise InputError(f"tts provider '{tts.family}' returned no voices")
    ctx.cache.put_json("voices", key, [voice.model_dump(mode="json") for voice in voices])
    return voices


def co_speech_matrix(scripts: Iterable[ChapterScript]) -> dict[str, set[str]]:
    """Characters that speak in the same paragraphs, per character (narration excluded)."""
    matrix: dict[str, set[str]] = defaultdict(set)
    for script in scripts:
        by_paragraph: dict[int, set[str]] = defaultdict(set)
        for segment in script.segments:
            if segment.kind == "dialogue" and segment.speaker != NARRATOR:
                by_paragraph[segment.paragraph_index].add(segment.speaker)
        for speakers in by_paragraph.values():
            for speaker in speakers:
                matrix[speaker].update(speakers - {speaker})
    return dict(matrix)


def stage_cast(ctx: JobContext) -> None:
    """Assign voices to the narrator and every speaking character; write ``cast.json``."""
    paths = ctx.paths
    if paths.cast.is_file() and not (paths.cast_overrides.is_file() and paths.cast_overrides.stat().st_mtime >= paths.cast.stat().st_mtime):
        ctx.log("info", "cast: cast.json exists; skipped")
        ctx.progress("cast", 1, 1, "cast.json present")
        return
    book = _load_book(ctx)
    bible = CastBible.model_validate(read_json(paths.bible))
    voices = _voice_catalog(ctx)
    write_json(paths.voices, [voice.model_dump(mode="json") for voice in voices])

    overrides: dict[str, str] = dict(ctx.job.options.cast_overrides)
    if paths.cast_overrides.is_file():
        stored = read_json(paths.cast_overrides)
        if not isinstance(stored, dict):
            raise InputError("cast_overrides.json must be a JSON object of character -> voice id")
        overrides.update({str(name): str(voice_id) for name, voice_id in stored.items()})
    scripts = [_load_script(ctx, chapter.index) for chapter in book.chapters]
    try:
        cast = cast_voices(bible, voices, ctx.providers.tts.family, overrides, ctx.job.source_sha256, co_speech_matrix(scripts))
    except ValueError as exc:
        raise InputError(str(exc), unit="cast") from exc
    write_json(paths.cast, cast)
    ctx.log("info", "cast: narrator " + cast.narrator.voice.id + "; " + ", ".join(f"{a.character} -> {a.voice.id}" for a in cast.characters))
    ctx.progress("cast", 1, 1, f"{len(cast.characters)} characters cast")


# --------------------------------------------------------------------------- render: tts
def split_text(text: str, max_chars: int) -> list[str]:
    """Split *text* into pieces of at most *max_chars* at sentence boundaries (then at spaces)."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        while len(sentence) > max_chars:                       # one giant sentence: cut at a space
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)
    return [piece for piece in pieces if piece]


def _neighbour_text(script: ChapterScript, index: int, direction: int) -> str | None:
    """Text of the nearest segment in *direction* spoken by the same speaker: within the
    paragraph for characters, across paragraphs for the narrator."""
    segment = script.segments[index]
    position = index + direction
    while 0 <= position < len(script.segments):
        other = script.segments[position]
        if segment.speaker != NARRATOR and other.paragraph_index != segment.paragraph_index:
            return None
        if other.speaker == segment.speaker:
            return other.text
        position += direction
    return None


def _tail(text: str | None) -> str | None:
    return text[-NEIGHBOUR_TEXT_CHARS:] if text else None


def _head(text: str | None) -> str | None:
    return text[:NEIGHBOUR_TEXT_CHARS] if text else None


def plan_tts(script: ChapterScript, cast: Cast, family: str, cache_version: str, max_chars: int) -> list[TtsJob]:
    """One TtsJob per (segment, piece) with its request and cache key."""
    jobs: list[TtsJob] = []
    for index, segment in enumerate(script.segments):
        assignment = cast.assignment_for(segment.speaker)
        settings = settings_for(assignment, segment.emotion, segment.delivery)
        pieces = split_text(segment.text, max_chars)
        before = _neighbour_text(script, index, -1)
        after = _neighbour_text(script, index, +1)
        for piece_index, piece in enumerate(pieces):
            previous = pieces[piece_index - 1] if piece_index > 0 else before
            following = pieces[piece_index + 1] if piece_index + 1 < len(pieces) else after
            request = TTSRequest(
                text=piece, voice_id=assignment.voice.id, settings=settings, emotion=segment.emotion,
                delivery=segment.delivery, previous_text=_tail(previous), next_text=_head(following), seed=assignment.seed,
            )
            jobs.append(TtsJob(segment_id=segment.id, piece=piece_index, request=request, clip_key=clip_key("tts", family, cache_version, request)))
    return jobs


def _render_tts(ctx: JobContext, script: ChapterScript, cast: Cast, chapter_no: int, chapter_count: int) -> ChapterTts:
    """Synthesize every cache miss of the chapter in parallel and record measured durations."""
    tts = ctx.providers.tts
    paths = ctx.paths
    index = script.chapter_index
    jobs = plan_tts(script, cast, tts.family, str(tts.cache_version), tts.max_chars)
    durations: dict[str, int] = {}
    if paths.tts(index).is_file():                      # durations of unchanged clips from the last run
        previous = ChapterTts.model_validate(read_json(paths.tts(index)))
        durations.update({key: ms for key, ms in previous.durations_ms.items() if ctx.cache.has("tts", key)})
    misses: list[TtsJob] = []
    seen: set[str] = set()
    for job in jobs:
        if job.clip_key in seen:
            continue
        seen.add(job.clip_key)
        if job.clip_key not in durations:
            cached = ctx.cache.get("tts", job.clip_key)
            if cached is None:
                misses.append(job)
                continue
            durations[job.clip_key] = cached.duration_ms
        _record_hit(ctx, "tts", job.clip_key, tts.family, "characters", float(len(job.request.text)), {"segment": job.segment_id})
    total = len(jobs)
    label = f"ch {chapter_no}/{chapter_count} tts"
    ctx.progress("tts", chapter_no - 1, chapter_count, f"{label} {total - len(misses)}/{total}")

    def synthesize(job: TtsJob) -> int:
        with FamilyLimiter.acquire(tts.family, ctx.settings.concurrency):
            clip = with_retry(lambda: tts.synthesize(job.request), on_retry=_retry_logger(ctx, _unit(index, job.segment_id)))
        finished = _finish_clip(clip, trim=True)
        ctx.cache.put("tts", job.clip_key, finished)
        ctx.mark_produced("tts", job.clip_key)
        return finished.duration_ms

    completed = total - len(misses)

    def record(job: TtsJob, duration: int) -> None:
        nonlocal completed
        durations[job.clip_key] = duration
        completed += 1
        ctx.progress("tts", chapter_no - 1, chapter_count, f"{label} {completed}/{total}")

    _run_parallel(ctx, misses, synthesize, lambda job: _unit(index, job.segment_id), record)
    result = ChapterTts(chapter_index=index, jobs=jobs, durations_ms={job.clip_key: durations[job.clip_key] for job in jobs})
    write_json(paths.tts(index), result)
    return result


# --------------------------------------------------------------------------- render: timeline, music, sfx, mix
def timeline_params(ctx: JobContext) -> TimelineParams:
    """TimelineParams from the selected providers, the job options and the settings."""
    music = ctx.providers.music
    sfx = ctx.providers.sfx
    return TimelineParams(
        music_enabled=ctx.job.options.music,
        sfx_enabled=ctx.job.options.sfx,
        music_family=music.family,
        music_cache_version=str(music.cache_version),
        music_min_ms=int(music.min_duration_ms),
        music_max_ms=int(music.max_duration_ms),
        sfx_family=sfx.family,
        sfx_cache_version=str(sfx.cache_version),
        sfx_max_ms=int(sfx.max_duration_ms),
        music_gain_db=ctx.settings.music_gain_db,
        sfx_gain_db=ctx.settings.sfx_gain_db,
        seed_material=ctx.job.source_sha256,
    )


def render_key(ctx: JobContext, script: ChapterScript, cast: Cast, params: TimelineParams) -> str:
    """Everything a chapter's outputs depend on; stored in the ``render.key`` sidecar."""
    tts = ctx.providers.tts
    return content_key(
        "render", SCHEMA_VERSION, script.model_dump(mode="json"), cast.model_dump(mode="json"),
        dataclasses.asdict(params), tts.family, str(tts.cache_version), ctx.settings.mp3,
    )


def _chapter_rendered(ctx: JobContext, chapter_index: int, key: str) -> bool:
    paths = ctx.paths
    sidecar = paths.render_key(chapter_index)
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != key:
        return False
    return all(path.is_file() for path in paths.chapter_outputs(chapter_index))


def _render_music(ctx: JobContext, timeline: ChapterTimeline, chapter_no: int, chapter_count: int) -> None:
    """Compose every missing music bed, sequentially (local models are memory-bound)."""
    music = ctx.providers.music
    index = timeline.chapter_index
    jobs = timeline.music_jobs
    label = f"ch {chapter_no}/{chapter_count} music"
    for position, job in enumerate(jobs):
        ctx.check_cancelled()
        seconds = job.request.duration_ms / 1000.0
        if ctx.cache.has("music", job.clip_key):
            _record_hit(ctx, "music", job.clip_key, music.family, "audio_seconds", seconds, {"cue": job.cue_id})
        else:
            unit = _unit(index, job.cue_id)
            try:
                with FamilyLimiter.acquire(music.family, ctx.settings.concurrency):
                    clip = with_retry(lambda: music.compose(job.request), on_retry=_retry_logger(ctx, unit))
                ctx.cache.put("music", job.clip_key, _finish_clip(clip, trim=False))
                ctx.mark_produced("music", job.clip_key)
            except BaseException as exc:
                _attach_unit(exc, unit)
                raise
        ctx.progress("music", chapter_no - 1, chapter_count, f"{label} {position + 1}/{len(jobs)}")
    ctx.progress("music", chapter_no - 1, chapter_count, f"{label} {len(jobs)}/{len(jobs)}")


def _render_sfx(ctx: JobContext, timeline: ChapterTimeline, chapter_no: int, chapter_count: int) -> None:
    """Generate every missing sound effect in parallel."""
    sfx = ctx.providers.sfx
    index = timeline.chapter_index
    jobs = timeline.sfx_jobs
    misses: list[SfxJob] = []
    for job in jobs:
        if ctx.cache.has("sfx", job.clip_key):
            _record_hit(ctx, "sfx", job.clip_key, sfx.family, "audio_seconds", job.request.duration_ms / 1000.0, {"cue": job.cue_id})
        else:
            misses.append(job)
    label = f"ch {chapter_no}/{chapter_count} sfx"
    completed = len(jobs) - len(misses)
    ctx.progress("sfx", chapter_no - 1, chapter_count, f"{label} {completed}/{len(jobs)}")

    def generate(job: SfxJob) -> None:
        with FamilyLimiter.acquire(sfx.family, ctx.settings.concurrency):
            clip = with_retry(lambda: sfx.generate(job.request), on_retry=_retry_logger(ctx, _unit(index, job.cue_id)))
        ctx.cache.put("sfx", job.clip_key, _finish_clip(clip, trim=False))
        ctx.mark_produced("sfx", job.clip_key)

    def record(job: SfxJob, _: None) -> None:
        nonlocal completed
        completed += 1
        ctx.progress("sfx", chapter_no - 1, chapter_count, f"{label} {completed}/{len(jobs)}")

    _run_parallel(ctx, misses, generate, lambda job: _unit(index, job.cue_id), record)


def _mix_chapter(ctx: JobContext, script: ChapterScript, cast: Cast, timeline: ChapterTimeline, key: str) -> ChapterManifest:
    """Mix the chapter, export mp3 when allowed, write ``manifest.json`` and the render key sidecar."""
    paths = ctx.paths
    index = script.chapter_index
    kinds = {placement.clip_key: TRACK_KIND[placement.track] for placement in timeline.placements}

    def load_clip(clip_key_: str) -> AudioClip:
        clip = ctx.cache.get(kinds[clip_key_], clip_key_)
        if clip is None:
            raise RenderError(f"clip {clip_key_} missing from the {kinds[clip_key_]} cache", unit=_unit(index, "mix"))
        return clip

    out_dir = paths.chapter_dir(index)
    written = mix_chapter(timeline, load_clip, out_dir)
    files: dict[str, str | None] = {name: paths.relative(path) for name, path in written.items()}
    files["mp3"] = None
    if ctx.settings.mp3 != "off" and export_mp3(written["mix"], paths.mp3(index)):
        files["mp3"] = paths.relative(paths.mp3(index))
    manifest = build_chapter_manifest(script, cast, timeline, files, index, script.title)
    write_json(paths.chapter_manifest(index), manifest)
    paths.render_key(index).write_text(key, encoding="utf-8")
    return manifest


def stage_render(ctx: JobContext) -> None:
    """Per chapter in order: tts -> timeline -> music -> sfx -> mix (+ chapter manifest)."""
    paths = ctx.paths
    book = _load_book(ctx)
    cast = _load_cast(ctx)
    params = timeline_params(ctx)
    chapter_count = len(book.chapters)
    max_ms = ctx.settings.max_chapter_minutes * 60_000
    ctx.progress("chapter", 0, chapter_count, f"ch 0/{chapter_count}")
    for chapter_no, chapter in enumerate(book.chapters, start=1):
        ctx.check_cancelled(force=True)
        index = chapter.index
        script = _load_script(ctx, index)
        key = render_key(ctx, script, cast, params)
        if _chapter_rendered(ctx, index, key):
            ctx.log("info", f"chapter {chapter_no}/{chapter_count} already rendered; skipped")
            ctx.progress("chapter", chapter_no, chapter_count, f"ch {chapter_no}/{chapter_count} cached")
            continue
        with ctx.substage(f"ch{index:02d} tts"):
            tts = _render_tts(ctx, script, cast, chapter_no, chapter_count)
        with ctx.substage(f"ch{index:02d} timeline"):
            ctx.progress("timeline", chapter_no - 1, chapter_count, f"ch {chapter_no}/{chapter_count} timeline")
            timeline = build_timeline(script, cast, tts, params)
            if max_ms and timeline.duration_ms > max_ms:
                raise RenderError(
                    f"chapter {index} would run {timeline.duration_ms / 60000:.1f} min; the limit is "
                    f"{ctx.settings.max_chapter_minutes} min (BOOKREADER_MAX_CHAPTER_MINUTES)",
                    unit=_unit(index, "timeline"),
                )
            write_json(paths.timeline(index), timeline)
        with ctx.substage(f"ch{index:02d} music"):
            _render_music(ctx, timeline, chapter_no, chapter_count)
        with ctx.substage(f"ch{index:02d} sfx"):
            _render_sfx(ctx, timeline, chapter_no, chapter_count)
        ctx.check_cancelled(force=True)
        with ctx.substage(f"ch{index:02d} mix"):
            ctx.progress("mix", chapter_no - 1, chapter_count, f"ch {chapter_no}/{chapter_count} mix")
            manifest = _mix_chapter(ctx, script, cast, timeline, key)
        ctx.log("info", f"chapter {chapter_no}/{chapter_count} mixed: {manifest.duration_ms} ms -> {manifest.files['mix']}")
        ctx.progress("chapter", chapter_no, chapter_count, f"ch {chapter_no}/{chapter_count} done")


# --------------------------------------------------------------------------- finalize
def stage_finalize(ctx: JobContext) -> None:
    """Write ``usage.json`` and ``manifest.json`` (last), then prune the shared cache."""
    ctx.check_cancelled(force=True)
    paths = ctx.paths
    book = _load_book(ctx)
    job = ctx.store.get_job(ctx.job.id) or ctx.job
    manifests: list[ChapterManifest] = []
    warnings: list[str] = []
    for chapter in book.chapters:
        manifests.append(ChapterManifest.model_validate(read_json(paths.chapter_manifest(chapter.index))))
        warnings.extend(f"ch{chapter.index:02d}: {warning}" for warning in _load_script(ctx, chapter.index).warnings)
    ctx.ledger.write_json(paths.usage)
    manifest = build_job_manifest(job, book, manifests, ctx.providers.describe(), warnings)
    write_json(paths.manifest, manifest)
    removed = ctx.cache.prune(ctx.settings.cache_max_mb * MB)
    if removed:
        ctx.log("info", f"finalize: pruned {len(removed)} cache file(s) to stay under {ctx.settings.cache_max_mb} MB")
    ctx.log("info", f"finalize: manifest.json written ({len(manifests)} chapter(s), {manifest.total_duration_ms} ms)")
    ctx.progress("finalize", 1, 1, "manifest.json written")


STAGE_FUNCTIONS: dict[Stage, Callable[[JobContext], None]] = {
    Stage.ingest: stage_ingest,
    Stage.analyze: stage_analyze,
    Stage.cast: stage_cast,
    Stage.render: stage_render,
    Stage.finalize: stage_finalize,
}
