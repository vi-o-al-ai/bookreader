"""bookreader.audio.timeline - the pure pacing and placement function of the render stage.

``build_timeline`` turns a ChapterScript, the cast and the measured TTS clip durations into a
ChapterTimeline: sequential voice placements with pacing pauses, word-anchored SFX (with a
blocking-gap mode for loud impacts that open a sentence), looped ambient beds, overlapping music
regions, and the MusicJob/SfxJob requests whose clip keys the cache and the mix stage share.
It performs no I/O and uses no randomness beyond ``stable_seed``.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

from bookreader.types import (
    ChapterScript,
    ChapterTimeline,
    ChapterTts,
    Cast,
    MusicCue,
    MusicJob,
    MusicRequest,
    Placement,
    Segment,
    SegmentTiming,
    SfxCue,
    SfxJob,
    SfxRequest,
    TtsJob,
    clip_key,
    stable_seed,
)

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- pacing table (ms)
INTRO_MS = 500                    # silence before the first word
MUSIC_INTRO_MS = 1500             # music-only intro when music is enabled
PIECE_GAP_MS = 120                # between the pieces of one split segment
PAUSE_SAME_MS = 250               # same speaker, same paragraph
PAUSE_SPEAKER_CHANGE_MS = 400
PAUSE_PARAGRAPH_MS = 650
PAUSE_SCENE_BREAK_MS = 1500
PAUSE_MIN_MS = 150
PAUSE_MAX_MS = 2500
EMOTION_PAUSE_MULTIPLIER: dict[str, float] = {"urgent": 0.6, "hesitant": 1.3, "melancholy": 1.3, "weary": 1.3}
VOICE_FADE_MS = 10
OUTRO_MS = 2500                   # tail after the last voice/sfx end

# --------------------------------------------------------------------------- sfx rules
SFX_ONSET_LEAD_MS = 120           # impacts start slightly before their anchor word
BLOCKING_INTENSITY = 0.7
BLOCKING_MAX_OFFSET_FRACTION = 0.25
BLOCKING_GAP_FRACTION = 0.6
BLOCKING_GAP_MAX_MS = 1500
IMPACT_FADE_IN_MS = 20
IMPACT_FADE_OUT_MS = 200
IMPACT_GAIN_MIN_DB = -20.0
IMPACT_GAIN_MAX_DB = 0.0
AMBIENT_LEAD_MS = 300
AMBIENT_GAIN_OFFSET_DB = -10.0
AMBIENT_FADE_IN_MS = 800
AMBIENT_FADE_OUT_MS = 1500
AMBIENT_REQUEST_MS = 15000
SFX_MIN_REQUEST_MS = 500

# --------------------------------------------------------------------------- music rules
MUSIC_REGION_LEAD_MS = 800
MUSIC_REGION_TAIL_MS = 1500
MUSIC_OVERLAP_MS = 3000
MUSIC_FADE_IN_MS = 2500
MUSIC_FIRST_FADE_IN_MS = 1500
MUSIC_FADE_OUT_MS = 3000
MUSIC_REQUEST_PAD_MS = 4000
MUSIC_REQUEST_CAP_MS = 120000

DEFAULT_MUSIC_PROMPTS: dict[str, str] = {
    "calm": "soft ambient pad, slow and gentle, warm piano touches, unobtrusive underscore",
    "warm": "warm acoustic guitar and strings, gentle and homely, light and comforting underscore",
    "tense": "low pulsing drone with tremolo strings, tension building, dark suspense underscore",
    "ominous": "dark low cluster drone, distant rumble, slow ominous swell, unsettling underscore",
    "melancholy": "slow minor piano arpeggio with soft strings, wistful and reflective underscore",
    "sad": "sparse minor pad and lonely cello, slow and mournful, sorrowful underscore",
    "hopeful": "rising major arpeggio with light strings, gentle optimism, uplifting underscore",
    "adventurous": "driving rhythmic strings and percussion, bold and energetic, adventurous underscore",
    "joyful": "bright staccato strings and playful woodwinds, cheerful and lively underscore",
    "mysterious": "detuned fifths and shimmering bells, curious and mysterious, subtle underscore",
    "romantic": "lush major ninth pad with soft harp, tender and intimate, romantic underscore",
    "none": "silence",
}


def default_music_prompt(mood: str) -> str:
    """Provider prompt used for a music cue that carries no prompt of its own."""
    return DEFAULT_MUSIC_PROMPTS.get(mood, DEFAULT_MUSIC_PROMPTS["calm"])


@dataclass(frozen=True)
class TimelineParams:
    """Everything build_timeline needs from settings and the selected providers."""

    music_enabled: bool
    sfx_enabled: bool
    music_family: str
    music_cache_version: str
    music_min_ms: int
    music_max_ms: int
    sfx_family: str
    sfx_cache_version: str
    sfx_max_ms: int
    music_gain_db: float
    sfx_gain_db: float
    seed_material: str


# --------------------------------------------------------------------------- helpers
def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def pause_before_ms(prev: Segment, seg: Segment) -> int:
    """Pacing pause inserted before *seg* when it follows *prev* (scene break > paragraph > speaker > same)."""
    if seg.scene_break_before:
        base = PAUSE_SCENE_BREAK_MS
    elif seg.paragraph_index != prev.paragraph_index:
        base = PAUSE_PARAGRAPH_MS
    elif seg.speaker != prev.speaker:
        base = PAUSE_SPEAKER_CHANGE_MS
    else:
        base = PAUSE_SAME_MS
    scaled = base * EMOTION_PAUSE_MULTIPLIER.get(seg.emotion, 1.0)
    return int(_clamp(round(scaled), PAUSE_MIN_MS, PAUSE_MAX_MS))


def is_blocking_impact(cue: SfxCue, seg: Segment) -> bool:
    """True when a loud impact opens a narration sentence and must precede the words instead of overlapping them."""
    if cue.kind != "impact" or cue.intensity < BLOCKING_INTENSITY or seg.kind != "narration":
        return False
    length = len(seg.text)
    fraction = cue.anchor_offset / length if length else 0.0
    return fraction < BLOCKING_MAX_OFFSET_FRACTION


def sfx_request(cue: SfxCue, params: TimelineParams) -> SfxRequest:
    """Provider request for a cue: durations clamped to the provider limit, seed from the description."""
    if cue.kind == "ambient":
        duration = min(AMBIENT_REQUEST_MS, params.sfx_max_ms)
    else:
        duration = int(_clamp(cue.duration_ms, SFX_MIN_REQUEST_MS, params.sfx_max_ms))
    return SfxRequest(
        description=cue.description,
        kind=cue.kind,
        duration_ms=duration,
        loop=cue.kind == "ambient",
        intensity=cue.intensity,
        seed=stable_seed(params.seed_material, cue.description),
    )


def music_request(cue: MusicCue, region_ms: int, params: TimelineParams) -> MusicRequest:
    """Provider request for a music region; the seed depends only on (book, mood, prompt) so beds are reused."""
    prompt = cue.prompt or default_music_prompt(cue.mood)
    duration = int(_clamp(min(region_ms + MUSIC_REQUEST_PAD_MS, MUSIC_REQUEST_CAP_MS), params.music_min_ms, params.music_max_ms))
    return MusicRequest(
        prompt=prompt,
        mood=cue.mood,
        energy=cue.energy,
        duration_ms=duration,
        loopable=True,
        seed=stable_seed(params.seed_material, cue.mood, prompt),
    )


def impact_gain_db(intensity: float, sfx_gain_db: float) -> float:
    """Impact level: the sfx bed level shifted by intensity, clamped to [-20, 0] dB."""
    return _clamp(sfx_gain_db + (intensity - BLOCKING_INTENSITY) * 10.0, IMPACT_GAIN_MIN_DB, IMPACT_GAIN_MAX_DB)


def _anchor_ms(cue: SfxCue, timing: SegmentTiming, seg: Segment) -> int:
    """Proportional position of the cue's anchor word inside the segment's measured span."""
    length = len(seg.text)
    fraction = _clamp(cue.anchor_offset / length, 0.0, 1.0) if length else 0.0
    return int(round(timing.start_ms + fraction * (timing.end_ms - timing.start_ms)))


# --------------------------------------------------------------------------- the pure function
def build_timeline(script: ChapterScript, cast: Cast, tts: ChapterTts, params: TimelineParams) -> ChapterTimeline:
    """Lay out one chapter: sequential voice, anchored sfx, looped ambients and overlapping music regions.

    *cast* is accepted for signature stability (the manifest uses it); placement does not depend on voices.
    """
    del cast  # placement is voice-agnostic; kept in the signature for the pipeline contract
    jobs_by_segment: dict[str, list[TtsJob]] = defaultdict(list)
    for job in tts.jobs:
        jobs_by_segment[job.segment_id].append(job)
    for jobs in jobs_by_segment.values():
        jobs.sort(key=lambda j: j.piece)

    segments_by_id = {seg.id: seg for seg in script.segments}
    sfx_by_span: dict[str, list[SfxCue]] = defaultdict(list)
    if params.sfx_enabled:
        for cue in script.sfx:
            if cue.span_id in segments_by_id:
                sfx_by_span[cue.span_id].append(cue)
            else:
                log.warning("chapter %d: sfx cue %s anchors unknown span %s; dropped", script.chapter_index, cue.id, cue.span_id)

    voice: list[Placement] = []
    timings: list[SegmentTiming] = []
    blocking_starts: dict[str, int] = {}
    cursor = MUSIC_INTRO_MS if params.music_enabled else INTRO_MS
    prev: Segment | None = None

    for seg in script.segments:
        if prev is not None:
            cursor += pause_before_ms(prev, seg)
        for cue in sfx_by_span.get(seg.id, []):
            if is_blocking_impact(cue, seg):
                blocking_starts[cue.id] = cursor
                cursor += int(min(BLOCKING_GAP_FRACTION * cue.duration_ms, BLOCKING_GAP_MAX_MS))
        start = cursor
        for index, job in enumerate(jobs_by_segment.get(seg.id, [])):
            if index:
                cursor += PIECE_GAP_MS
            duration = tts.durations_ms.get(job.clip_key)
            if duration is None:
                log.warning("chapter %d: no measured duration for %s piece %d; placing 0 ms", script.chapter_index, seg.id, job.piece)
                duration = 0
            voice.append(Placement(
                track="voice", clip_key=job.clip_key, ref_id=seg.id, start_ms=cursor, end_ms=cursor + duration,
                gain_db=0.0, fade_in_ms=VOICE_FADE_MS, fade_out_ms=VOICE_FADE_MS,
            ))
            cursor += duration
        timings.append(SegmentTiming(id=seg.id, start_ms=start, end_ms=cursor))
        prev = seg

    timing_by_id = {t.id: t for t in timings}
    sfx: list[Placement] = []
    sfx_jobs: dict[str, SfxJob] = {}
    for cue in script.sfx:
        seg = segments_by_id.get(cue.span_id)
        if not params.sfx_enabled or seg is None:
            continue
        timing = timing_by_id[cue.span_id]
        request = sfx_request(cue, params)
        key = clip_key("sfx", params.sfx_family, params.sfx_cache_version, request)
        sfx_jobs.setdefault(key, SfxJob(cue_id=cue.id, request=request, clip_key=key))
        if cue.kind == "ambient":
            end_timing = timing_by_id.get(cue.end_span_id or "", timing)
            start_ms = max(0, _anchor_ms(cue, timing, seg) - AMBIENT_LEAD_MS)
            end_ms = max(start_ms, end_timing.end_ms)
            sfx.append(Placement(
                track="sfx", clip_key=key, ref_id=cue.id, start_ms=start_ms, end_ms=end_ms,
                gain_db=params.sfx_gain_db + AMBIENT_GAIN_OFFSET_DB, fade_in_ms=AMBIENT_FADE_IN_MS,
                fade_out_ms=AMBIENT_FADE_OUT_MS, loop=True, duck="ambient",
            ))
            continue
        if cue.id in blocking_starts:
            start_ms = blocking_starts[cue.id]
        else:
            start_ms = max(0, _anchor_ms(cue, timing, seg) - SFX_ONSET_LEAD_MS)
        sfx.append(Placement(
            track="sfx", clip_key=key, ref_id=cue.id, start_ms=start_ms, end_ms=start_ms + request.duration_ms,
            gain_db=impact_gain_db(cue.intensity, params.sfx_gain_db), fade_in_ms=IMPACT_FADE_IN_MS,
            fade_out_ms=IMPACT_FADE_OUT_MS, loop=False, duck="none",
        ))

    last_end = max([cursor] + [p.end_ms for p in sfx])
    duration_ms = last_end + OUTRO_MS

    music: list[Placement] = []
    music_jobs: dict[str, MusicJob] = {}
    if params.music_enabled:
        regions: list[tuple[MusicCue, int, int]] = []
        for cue in script.music:
            start_t = timing_by_id.get(cue.start_span_id)
            end_t = timing_by_id.get(cue.end_span_id)
            if start_t is None or end_t is None:
                log.warning("chapter %d: music cue %s references unknown spans; dropped", script.chapter_index, cue.id)
                continue
            start_ms = int(_clamp(start_t.start_ms - MUSIC_REGION_LEAD_MS, 0, duration_ms))
            end_ms = int(_clamp(end_t.end_ms + MUSIC_REGION_TAIL_MS, start_ms, duration_ms))
            regions.append((cue, start_ms, end_ms))
        for index in range(1, len(regions)):
            cue, start_ms, _ = regions[index]
            prev_cue, prev_start, _ = regions[index - 1]
            regions[index - 1] = (prev_cue, prev_start, int(_clamp(start_ms + MUSIC_OVERLAP_MS, prev_start, duration_ms)))
        for index, (cue, start_ms, end_ms) in enumerate(regions):
            request = music_request(cue, end_ms - start_ms, params)
            key = clip_key("music", params.music_family, params.music_cache_version, request)
            music_jobs.setdefault(key, MusicJob(cue_id=cue.id, request=request, clip_key=key))
            music.append(Placement(
                track="music", clip_key=key, ref_id=cue.id, start_ms=start_ms, end_ms=end_ms,
                gain_db=params.music_gain_db, fade_in_ms=MUSIC_FIRST_FADE_IN_MS if index == 0 else MUSIC_FADE_IN_MS,
                fade_out_ms=MUSIC_FADE_OUT_MS, loop=True, duck="music",
            ))

    return ChapterTimeline(
        chapter_index=script.chapter_index,
        duration_ms=duration_ms,
        segments=timings,
        placements=voice + sfx + music,
        music_jobs=list(music_jobs.values()),
        sfx_jobs=list(sfx_jobs.values()),
    )
