"""bookreader.audio.mixer - renders a ChapterTimeline into stems and a mastered mix.

``mix_chapter`` allocates one float32 buffer per track (voice, music, sfx), places every clip,
shapes the music/ambient beds around speech, masters the sum with a soft limiter and writes
``mix.wav`` plus the three stems (all identical length). The stems are the post-gain, post-duck
tracks, so ``voice + music + sfx`` equals the mix exactly whenever the limiter did not engage.

Level design (spec MIXER): the music bed sits at ``settings.music_gain_db`` under speech (the
beds are RMS-normalized to -20 dBFS in the render stage, so -14 dB puts them at -34 dBFS under a
-20 dBFS voice) and rises ``MUSIC_GAP_BOOST_DB`` in speech-free gaps of at least
``MUSIC_GAP_MIN_MS``; gaps the timeline opened for a blocking impact are not boosted. Ambient beds
dip ``AMBIENT_DUCK_DB`` under speech; impacts are never ducked.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from bookreader.types import (
    SAMPLE_RATE,
    AudioClip,
    Cast,
    ChapterManifest,
    ChapterScript,
    ChapterTimeline,
    ManifestCue,
    ManifestSegment,
    Placement,
)

from bookreader.audio.dsp import db_to_gain, fade, loop_to_length, ms_to_samples, rms_envelope, soft_limit
from bookreader.audio.pcm import resample, to_float, to_int16, write_wav

log = logging.getLogger(__name__)

TRACKS: tuple[str, ...] = ("voice", "music", "sfx")
DUCK_FRAME_MS = 20
SPEECH_THRESHOLD_DBFS = -45.0
DUCK_ATTACK_MS = 40
DUCK_RELEASE_MS = 600
DUCK_LOOKAHEAD_MS = 60       # the gain starts dropping this early so speech onsets are not smeared over
MUSIC_DUCK_DB = 0.0          # under speech the bed follows music_gain_db exactly (no extra attenuation)
MUSIC_GAP_BOOST_DB = 5.0     # music rises this much in gaps longer than MUSIC_GAP_MIN_MS
MUSIC_GAP_MIN_MS = 1200
AMBIENT_DUCK_DB = 4.0        # ambient dip under speech
LIMIT_KNEE_DBFS = -3.0
LIMIT_CEILING_DBFS = -1.0


# --------------------------------------------------------------------------- ducking
def _gap_runs(present: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) frame ranges where speech is absent."""
    if len(present) == 0:
        return []
    absent = np.concatenate([[False], ~present, [False]]).astype(np.int8)
    edges = np.diff(absent)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def duck_gain(
    voice_env: np.ndarray,
    n_samples: int,
    depth_db: float,
    attack_ms: int = DUCK_ATTACK_MS,
    release_ms: int = DUCK_RELEASE_MS,
    *,
    gap_boost_db: float = 0.0,
    min_gap_ms: int = 0,
    lookahead_ms: int = DUCK_LOOKAHEAD_MS,
    hold: np.ndarray | None = None,
    frame_ms: int = DUCK_FRAME_MS,
    threshold_dbfs: float = SPEECH_THRESHOLD_DBFS,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Per-sample gain curve driven by the voice envelope (one value per *frame_ms* frame).

    Under speech, and in gaps shorter than *min_gap_ms*, the gain is ``-depth_db``; in longer gaps it is
    ``+gap_boost_db``. Speech presence is extended *lookahead_ms* earlier so the drop is complete by
    the onset. Frames flagged in *hold* (one bool per frame, e.g. an impact sound playing in a gap the
    timeline opened for it) count as busy: they stay at the speech gain and never earn the gap boost.
    Transitions are one-pole smoothed: *attack_ms* when the gain drops (speech onset), *release_ms*
    when it rises. The frame curve is linearly interpolated to sample resolution.
    """
    n_frames = len(voice_env)
    speech_gain = db_to_gain(-abs(depth_db))
    gap_gain = db_to_gain(gap_boost_db)
    if n_frames == 0:
        return np.full(n_samples, gap_gain, dtype=np.float32)
    present = np.asarray(voice_env) > threshold_dbfs
    lead = -(-lookahead_ms // frame_ms) if lookahead_ms > 0 else 0
    for k in range(1, min(lead, n_frames - 1) + 1):
        present[:-k] |= present[k:]
    if hold is not None:
        held = np.asarray(hold, dtype=bool)[:n_frames]
        present[: len(held)] |= held
    target = np.full(n_frames, speech_gain, dtype=np.float64)
    min_gap_frames = -(-min_gap_ms // frame_ms) if min_gap_ms > 0 else 0
    for start, end in _gap_runs(present):
        if end - start >= min_gap_frames:
            target[start:end] = gap_gain
    attack = float(np.exp(-frame_ms / max(attack_ms, 1e-3)))
    release = float(np.exp(-frame_ms / max(release_ms, 1e-3)))
    smoothed = np.empty(n_frames, dtype=np.float64)
    level = target[0]
    for i, goal in enumerate(target):
        coef = attack if goal < level else release
        level = goal + (level - goal) * coef
        smoothed[i] = level
    frame = max(1, ms_to_samples(frame_ms, sample_rate))
    centers = np.arange(n_frames, dtype=np.float64) * frame + frame / 2.0
    return np.interp(np.arange(n_samples, dtype=np.float64), centers, smoothed).astype(np.float32)


def duck(
    music: np.ndarray,
    voice_env: np.ndarray,
    depth_db: float,
    attack_ms: int = DUCK_ATTACK_MS,
    release_ms: int = DUCK_RELEASE_MS,
    *,
    gap_boost_db: float = 0.0,
    min_gap_ms: int = 0,
    frame_ms: int = DUCK_FRAME_MS,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Sidechain-duck *music* under the speech described by *voice_env* (see :func:`duck_gain`)."""
    x = to_float(music)
    curve = duck_gain(
        voice_env, len(x), depth_db, attack_ms, release_ms,
        gap_boost_db=gap_boost_db, min_gap_ms=min_gap_ms, frame_ms=frame_ms, sample_rate=sample_rate,
    )
    return (x * curve).astype(np.float32)


# --------------------------------------------------------------------------- rendering
def render_track(
    placements: list[Placement],
    load_clip: Callable[[str], AudioClip],
    n_samples: int,
    sample_rate: int = SAMPLE_RATE,
    duck_curves: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Sum *placements* into one float32 buffer of *n_samples*.

    Each clip is looped or trimmed to its slot, faded, gained and (when its ``duck`` kind has a curve in
    *duck_curves*) multiplied by that per-sample curve before being added.
    """
    out = np.zeros(n_samples, dtype=np.float32)
    clips: dict[str, np.ndarray] = {}
    for p in placements:
        start = ms_to_samples(p.start_ms, sample_rate)
        end = min(ms_to_samples(p.end_ms, sample_rate), n_samples)
        if start >= end or start < 0:
            log.debug("placement %s (%s) falls outside the chapter; skipped", p.ref_id, p.track)
            continue
        source = clips.get(p.clip_key)
        if source is None:
            clip = load_clip(p.clip_key)
            source = to_float(clip.samples)
            if clip.sample_rate != sample_rate:
                source = resample(source, clip.sample_rate, sample_rate)
            clips[p.clip_key] = source
        slot = end - start
        x = loop_to_length(source, slot, sample_rate=sample_rate) if p.loop else source[:slot]
        if len(x) == 0:
            continue
        x = fade(x, p.fade_in_ms, p.fade_out_ms, sample_rate) * np.float32(db_to_gain(p.gain_db))
        if duck_curves is not None and p.duck in duck_curves:
            x = x * duck_curves[p.duck][start: start + len(x)]
        out[start: start + len(x)] += x
    return out


def master(voice: np.ndarray, music: np.ndarray, sfx: np.ndarray) -> tuple[np.ndarray, bool]:
    """Sum the tracks; soft-limit when the peak exceeds the ceiling. Returns (mix, limiter_engaged)."""
    mix = (voice + music + sfx).astype(np.float32)
    peak = float(np.max(np.abs(mix))) if len(mix) else 0.0
    if peak > db_to_gain(LIMIT_CEILING_DBFS):
        log.info("mix peak %.3f above ceiling; soft limiter engaged", peak)
        return soft_limit(mix, LIMIT_KNEE_DBFS, LIMIT_CEILING_DBFS), True
    return mix, False


def mix_chapter(
    timeline: ChapterTimeline,
    load_clip: Callable[[str], AudioClip],
    out_dir: Path,
    sample_rate: int = SAMPLE_RATE,
) -> dict[str, Path]:
    """Render *timeline* into ``out_dir/{mix,voice,music,sfx}.wav`` and return their paths by name."""
    out_dir = Path(out_dir)
    n = ms_to_samples(timeline.duration_ms, sample_rate)
    by_track = {name: [p for p in timeline.placements if p.track == name] for name in TRACKS}

    voice = render_track(by_track["voice"], load_clip, n, sample_rate)
    voice_env = rms_envelope(voice, DUCK_FRAME_MS, sample_rate)
    # Un-ducked sfx (impacts) are rendered first: where one is sounding the music must not treat the
    # silence around it (typically a blocking gap the timeline opened for it) as breathing room.
    impacts = render_track([p for p in by_track["sfx"] if p.duck == "none"], load_clip, n, sample_rate)
    impact_present = rms_envelope(impacts, DUCK_FRAME_MS, sample_rate) > SPEECH_THRESHOLD_DBFS
    curves = {
        "music": duck_gain(
            voice_env, n, MUSIC_DUCK_DB, gap_boost_db=MUSIC_GAP_BOOST_DB, min_gap_ms=MUSIC_GAP_MIN_MS,
            hold=impact_present, sample_rate=sample_rate,
        ),
        "ambient": duck_gain(voice_env, n, AMBIENT_DUCK_DB, sample_rate=sample_rate),
    }
    music = render_track(by_track["music"], load_clip, n, sample_rate, curves)
    sfx = impacts + render_track([p for p in by_track["sfx"] if p.duck != "none"], load_clip, n, sample_rate, curves)

    stems = {"voice": to_int16(voice), "music": to_int16(music), "sfx": to_int16(sfx)}
    mixed, limited = master(voice, music, sfx)
    if limited:
        mix16 = to_int16(mixed)
    else:  # exact stem sum: the stems are the quantized tracks, so their sum is the pre-limiter mix
        total = stems["voice"].astype(np.int32) + stems["music"].astype(np.int32) + stems["sfx"].astype(np.int32)
        mix16 = np.clip(total, -32768, 32767).astype(np.int16)

    paths: dict[str, Path] = {}
    for name, samples in {"mix": mix16, **stems}.items():
        paths[name] = write_wav(out_dir / f"{name}.wav", AudioClip(samples, sample_rate))
    log.info("chapter %d mixed: %d ms, limiter=%s", timeline.chapter_index, timeline.duration_ms, limited)
    return paths


# --------------------------------------------------------------------------- manifest
def build_chapter_manifest(
    script: ChapterScript,
    cast: Cast,
    timeline: ChapterTimeline,
    files: dict[str, str | None],
    index: int,
    title: str,
) -> ChapterManifest:
    """Pure: combine script text, cast voices and timeline timings into the public ChapterManifest."""
    timing_by_id = {t.id: t for t in timeline.segments}
    music_cues = {c.id: c for c in script.music}
    sfx_cues = {c.id: c for c in script.sfx}
    music_prompts = {j.clip_key: j.request.prompt for j in timeline.music_jobs}

    segments: list[ManifestSegment] = []
    for seg in script.segments:
        timing = timing_by_id.get(seg.id)
        if timing is None:
            log.warning("chapter %d: segment %s has no timing; omitted from manifest", index, seg.id)
            continue
        segments.append(ManifestSegment(
            id=seg.id, speaker=seg.speaker, voice_id=cast.assignment_for(seg.speaker).voice.id, kind=seg.kind,
            text=seg.text, emotion=seg.emotion, delivery=seg.delivery, paragraph=seg.paragraph_index,
            start_ms=timing.start_ms, end_ms=timing.end_ms, source=seg.source,
        ))

    cues: list[ManifestCue] = []
    for p in timeline.placements:
        if p.track == "music":
            cue = music_cues.get(p.ref_id)
            cues.append(ManifestCue(
                id=p.ref_id, kind="music", start_ms=p.start_ms, end_ms=p.end_ms, gain_db=p.gain_db, clip=p.clip_key,
                mood=cue.mood if cue else None, prompt=music_prompts.get(p.clip_key, cue.prompt if cue else None),
            ))
        elif p.track == "sfx":
            cue = sfx_cues.get(p.ref_id)
            cues.append(ManifestCue(
                id=p.ref_id, kind="sfx", start_ms=p.start_ms, end_ms=p.end_ms, gain_db=p.gain_db, clip=p.clip_key,
                description=cue.description if cue else None, sfx_kind=cue.kind if cue else None,
                anchor_segment=cue.span_id if cue else None, anchor_text=cue.anchor_text if cue else None,
            ))

    return ChapterManifest(
        index=index, title=title, duration_ms=timeline.duration_ms, sample_rate=SAMPLE_RATE,
        files=dict(files), segments=segments, cues=cues,
    )
