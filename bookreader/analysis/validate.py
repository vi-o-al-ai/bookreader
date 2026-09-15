"""bookreader.analysis.validate - repair or reject one analyzer output for one chunk.

Repairs (each one is reported as a warning): labels for unknown or duplicate span ids are
dropped, quote spans without a label are labelled by the heuristic analyzer, speakers are
coerced to canonical names through the bible and this chunk's character updates, SFX durations
are clamped to 0.5..30 s and intensities/energies to 0..1. Two problems cannot be repaired and
raise :class:`AnalysisInvalid`: more than 20 % of the quote spans unlabelled, or a cue that
references a span outside the chunk.
"""
from __future__ import annotations

import logging

from bookreader.providers.mock.analysis import HeuristicAnalyzer
from bookreader.types import NARRATOR, CastBible, CharacterUpdate, Chunk, ChunkAnalysis, MusicCueRaw, SfxCueRaw, SpanLabel, normalize_name

log = logging.getLogger(__name__)

MAX_UNLABELLED_FRACTION = 0.2
MIN_SFX_SECONDS = 0.5
MAX_SFX_SECONDS = 30.0


class AnalysisInvalid(ValueError):
    """The analyzer output cannot be repaired for this chunk; the message names the offending ids."""


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def canonical_speaker(speaker: str, bible: CastBible, updates: list[CharacterUpdate]) -> str:
    """Resolve a label speaker to the bible's canonical name, or to the name this chunk's
    updates will register it under; unknown names pass through unchanged."""
    key = normalize_name(speaker)
    if not key or key == NARRATOR.lower():
        return NARRATOR
    entry = bible.find(speaker)
    if entry is not None:
        return entry.name
    for update in updates:
        if key == normalize_name(update.name) or key in {normalize_name(a) for a in update.aliases}:
            return update.name.strip() or speaker.strip()
    return speaker.strip()


def validate_chunk_analysis(analysis: ChunkAnalysis, chunk: Chunk, bible: CastBible) -> tuple[ChunkAnalysis, list[str]]:
    """Return ``(repaired analysis, repair warnings)``; the repaired analysis also carries the
    warnings appended to its own ``warnings`` list so they travel with cached results."""
    known = {span.id for span in chunk.spans}
    order = {span.id: index for index, span in enumerate(chunk.spans)}
    quote_ids = [span.id for span in chunk.spans if span.kind == "quote"]
    warnings: list[str] = []

    labels: dict[str, SpanLabel] = {}
    for label in analysis.labels:
        if label.span_id not in known:
            warnings.append(f"dropped label for unknown span {label.span_id}")
        elif label.span_id in labels:
            warnings.append(f"dropped duplicate label for span {label.span_id}")
        else:
            labels[label.span_id] = label

    missing = [span_id for span_id in quote_ids if span_id not in labels]
    if missing and len(missing) > MAX_UNLABELLED_FRACTION * len(quote_ids):
        raise AnalysisInvalid(f"{len(missing)} of {len(quote_ids)} quote spans have no label: {', '.join(missing)}")
    outside = [cue.span_id for cue in [*analysis.sfx_cues, *analysis.music_cues] if cue.span_id not in known]
    if outside:
        raise AnalysisInvalid(f"cues reference spans outside the chunk: {', '.join(outside)}")

    if missing:
        filled = {label.span_id: label for label in HeuristicAnalyzer().label_missing(chunk, bible, missing)}
        for span_id in missing:
            label = filled.get(span_id) or SpanLabel(span_id=span_id, speaker=NARRATOR)
            labels[span_id] = label
            warnings.append(f"{span_id}: no label from the analyzer; heuristic assigned {label.speaker!r}")

    repaired_labels: list[SpanLabel] = []
    for span_id in sorted(labels, key=order.__getitem__):
        label = labels[span_id]
        speaker = canonical_speaker(label.speaker, bible, analysis.characters)
        if speaker != label.speaker:
            warnings.append(f"{span_id}: speaker {label.speaker!r} coerced to {speaker!r}")
            label = label.model_copy(update={"speaker": speaker})
        repaired_labels.append(label)

    sfx: list[SfxCueRaw] = []
    for cue in analysis.sfx_cues:
        duration = _clamp(cue.duration_s, MIN_SFX_SECONDS, MAX_SFX_SECONDS)
        intensity = _clamp(cue.intensity, 0.0, 1.0)
        if duration != cue.duration_s or intensity != cue.intensity:
            warnings.append(f"{cue.span_id}: sfx cue clamped (duration {cue.duration_s} -> {duration}, intensity {cue.intensity} -> {intensity})")
            cue = cue.model_copy(update={"duration_s": duration, "intensity": intensity})
        sfx.append(cue)
    music: list[MusicCueRaw] = []
    for cue in analysis.music_cues:
        energy = _clamp(cue.energy, 0.0, 1.0)
        if energy != cue.energy:
            warnings.append(f"{cue.span_id}: music energy {cue.energy} clamped to {energy}")
            cue = cue.model_copy(update={"energy": energy})
        music.append(cue)

    for warning in warnings:
        log.debug("chunk %d/%d repair: %s", chunk.chapter_index, chunk.chunk_index, warning)
    repaired = analysis.model_copy(
        update={"labels": repaired_labels, "sfx_cues": sfx, "music_cues": music, "warnings": [*analysis.warnings, *warnings]}
    )
    return repaired, warnings
