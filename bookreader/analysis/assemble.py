"""bookreader.analysis.assemble - turn validated chunk analyses into one ChapterScript.

Segments correspond 1:1 to the chapter's spans in order (no title segment). SFX cues become
:class:`SfxCue` with chapter-scoped ids ``c{ch}x{nnn}``; music start/change/stop actions are
resolved left to right across chunks into contiguous :class:`MusicCue` regions ``c{ch}m{nnn}``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from bookreader.types import (
    NARRATOR,
    CastBible,
    Chapter,
    ChapterScript,
    ChunkAnalysis,
    Mood,
    MusicCue,
    Segment,
    SfxCue,
    Span,
    SpanLabel,
    normalize_name,
)

log = logging.getLogger(__name__)

SFX_ID_FORMAT = "c{chapter}x{n:03d}"
MUSIC_ID_FORMAT = "c{chapter}m{n:03d}"
DEFAULT_MOOD: Mood = "calm"
CARRY_ENERGY = 0.2               # energy of a region that only carries the mood already in force

MOOD_PROMPTS: dict[str, str] = {
    "calm": "soft ambient pad, slow and gentle, sparse piano, unobtrusive instrumental underscore",
    "warm": "warm acoustic guitar and low strings, homely and comforting, gentle instrumental underscore",
    "tense": "low pulsing drone with tremolo strings, building tension, dark suspense underscore, no vocals",
    "ominous": "deep sustained drones and distant percussion, foreboding, slow cinematic underscore",
    "melancholy": "slow solo cello and soft piano, wistful and sad, sparse instrumental underscore",
    "sad": "quiet piano and muted strings, mournful and slow, gentle instrumental underscore",
    "hopeful": "light strings and soft woodwinds rising gently, optimistic and airy instrumental underscore",
    "adventurous": "driving strings and light percussion, bright and forward-moving orchestral underscore",
    "joyful": "bright acoustic guitar and playful pizzicato, cheerful and light instrumental underscore",
    "mysterious": "shimmering pads, soft harp and sparse bells, curious and enigmatic instrumental underscore",
    "romantic": "tender piano and warm strings, intimate and slow, gentle instrumental underscore",
    "none": "",
}


@dataclass
class _Region:
    start: int
    mood: str
    energy: float
    prompt: str
    end: int = -1


def _resolve_speaker(label: SpanLabel | None, span: Span, bible: CastBible, warnings: list[str]) -> str:
    if label is None:
        warnings.append(f"{span.id}: quote span has no label; spoken by narrator")
        return NARRATOR
    if not label.speaker.strip() or normalize_name(label.speaker) == NARRATOR.lower():
        return NARRATOR
    entry = bible.find(label.speaker)
    if entry is None:
        warnings.append(f"{span.id}: speaker {label.speaker!r} is not in the cast bible; spoken by narrator")
        return NARRATOR
    return entry.name


def _ambient_end(spans: list[tuple[int, Span, bool]], start: int) -> str:
    """Id of the last span before the next scene break (or the chapter end) after *start*."""
    end = spans[-1][1].id
    for index, span, scene_break in spans[start + 1:]:
        if scene_break:
            end = spans[index - 1][1].id
            break
    return end


def _resolve_music(
    chapter_index: int,
    analyses: list[ChunkAnalysis],
    index_of: dict[str, int],
    span_count: int,
    prior_mood: Mood,
    warnings: list[str],
) -> list[MusicCue]:
    if span_count == 0:
        return []
    events = []
    for analysis in analyses:
        for cue in analysis.music_cues:
            if cue.span_id not in index_of:
                warnings.append(f"dropped music cue for unknown span {cue.span_id}")
                continue
            events.append((index_of[cue.span_id], cue))
    events.sort(key=lambda item: item[0])          # stable: cues on the same span keep analysis order

    regions: list[_Region] = []
    open_region: _Region | None = None
    if prior_mood != "none":
        open_region = _Region(0, prior_mood, CARRY_ENERGY, MOOD_PROMPTS.get(prior_mood, ""))

    def close(at: int) -> None:
        nonlocal open_region
        if open_region is not None and at >= open_region.start:
            open_region.end = at
            regions.append(open_region)
        open_region = None

    for index, cue in events:
        if cue.action == "stop":
            if open_region is None:
                warnings.append(f"{cue.span_id}: music stop with no region playing")
            close(index - 1)
            continue
        prompt = cue.prompt or MOOD_PROMPTS.get(cue.mood, "")
        if open_region is not None and open_region.start == index:
            open_region.mood, open_region.energy, open_region.prompt = cue.mood, cue.energy, prompt
            continue
        close(index - 1)
        open_region = _Region(index, cue.mood, cue.energy, prompt)
    close(span_count - 1)

    if not regions:
        regions.append(_Region(0, DEFAULT_MOOD, CARRY_ENERGY, MOOD_PROMPTS[DEFAULT_MOOD], span_count - 1))
    ids = list(index_of)
    return [
        MusicCue(
            id=MUSIC_ID_FORMAT.format(chapter=chapter_index, n=n),
            start_span_id=ids[region.start],
            end_span_id=ids[region.end],
            mood=region.mood,  # type: ignore[arg-type]
            energy=region.energy,
            prompt=region.prompt,
        )
        for n, region in enumerate(regions, start=1)
    ]


def assemble_script(
    chapter: Chapter,
    analyses: list[ChunkAnalysis],
    bible: CastBible,
    prior_mood: Mood = "none",
) -> ChapterScript:
    """Build the chapter's script from its chunk analyses (in chunk order) and the final bible.

    *prior_mood* is the music mood in force when the chapter begins (the analyze stage knows it
    from the previous chapter); it becomes the opening region when the chapter's first music
    action starts later than its first span, or the whole chapter's region when there are no
    music actions at all (``"none"`` -> a quiet ``calm`` bed).
    """
    warnings: list[str] = [w for analysis in analyses for w in analysis.warnings]
    ordered: list[tuple[int, Span, bool]] = []
    for paragraph in chapter.paragraphs:
        for position, span in enumerate(paragraph.spans):
            ordered.append((len(ordered), span, paragraph.scene_break_before and position == 0))
    index_of = {span.id: index for index, span, _ in ordered}

    labels: dict[str, tuple[SpanLabel, str]] = {}
    for analysis in analyses:
        for label in analysis.labels:
            if label.span_id not in index_of:
                warnings.append(f"dropped label for unknown span {label.span_id}")
            elif label.span_id in labels:
                warnings.append(f"dropped duplicate label for span {label.span_id}")
            else:
                labels[label.span_id] = (label, analysis.source)

    segments: list[Segment] = []
    current_source = analyses[0].source if analyses else "heuristic"
    for paragraph in chapter.paragraphs:
        for position, span in enumerate(paragraph.spans):
            found = labels.get(span.id)
            label = None
            if found is not None:
                label, current_source = found
            if span.kind == "narration":
                speaker, kind = NARRATOR, "narration"
            else:
                speaker, kind = _resolve_speaker(label, span, bible, warnings), "dialogue"
            segments.append(
                Segment(
                    id=span.id,
                    paragraph_index=paragraph.index,
                    speaker=speaker,
                    kind=kind,  # type: ignore[arg-type]
                    text=span.text,
                    emotion=label.emotion if label else "neutral",
                    delivery=label.delivery if label else "normal",
                    scene_break_before=paragraph.scene_break_before and position == 0,
                    source=current_source,  # type: ignore[arg-type]
                )
            )

    sfx: list[SfxCue] = []
    for analysis in analyses:
        for cue in analysis.sfx_cues:
            index = index_of.get(cue.span_id)
            if index is None:
                warnings.append(f"dropped sfx cue for unknown span {cue.span_id}")
                continue
            span = ordered[index][1]
            offset = span.text.lower().find(cue.anchor_text.lower()) if cue.anchor_text else -1
            if offset < 0:
                warnings.append(f"{cue.span_id}: sfx anchor {cue.anchor_text!r} not found in span; anchored at start")
                offset = 0
            sfx.append(
                SfxCue(
                    id=SFX_ID_FORMAT.format(chapter=chapter.index, n=len(sfx) + 1),
                    span_id=cue.span_id,
                    anchor_text=cue.anchor_text,
                    anchor_offset=offset,
                    description=cue.description,
                    kind=cue.kind,
                    duration_ms=int(round(cue.duration_s * 1000)),
                    intensity=cue.intensity,
                    end_span_id=_ambient_end(ordered, index) if cue.kind == "ambient" else None,
                )
            )

    music = _resolve_music(chapter.index, analyses, index_of, len(ordered), prior_mood, warnings)
    log.debug("chapter %d assembled: %d segments, %d sfx, %d music regions, %d warnings", chapter.index, len(segments), len(sfx), len(music), len(warnings))
    return ChapterScript(chapter_index=chapter.index, title=chapter.title, segments=segments, music=music, sfx=sfx, warnings=warnings)
