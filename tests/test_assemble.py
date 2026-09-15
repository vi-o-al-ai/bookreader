"""assemble_script: segments, sfx cues, music regions across chunks, scene-break handling."""
from __future__ import annotations

from pathlib import Path

import pytest

from bookreader.analysis.assemble import assemble_script
from bookreader.analysis.bible import apply_updates, finalize, register_speaker
from bookreader.analysis.chunker import make_chunks
from bookreader.analysis.validate import validate_chunk_analysis
from bookreader.ingest import load_book
from bookreader.providers.mock.analysis import HeuristicAnalyzer
from bookreader.types import (
    NARRATOR,
    CastBible,
    Chapter,
    CharacterEntry,
    ChapterScript,
    ChunkAnalysis,
    MusicCueRaw,
    Paragraph,
    SfxCueRaw,
    Span,
    SpanLabel,
)


def _chapter(*paragraphs: tuple[list[tuple[str, str]], bool], index: int = 1) -> Chapter:
    """Build a chapter from ([(kind, text), ...], scene_break_before) tuples."""
    out = []
    for p, (pieces, scene_break) in enumerate(paragraphs, start=1):
        spans, cursor = [], 0
        for s, (kind, text) in enumerate(pieces):
            spans.append(Span(id=f"c{index}p{p}s{s}", kind=kind, text=text, start_char=cursor, end_char=cursor + len(text)))  # type: ignore[arg-type]
            cursor += len(text) + 1
        out.append(Paragraph(index=p, text=" ".join(t for _, t in pieces), scene_break_before=scene_break, spans=spans))
    return Chapter(index=index, title=f"Chapter {index}", paragraphs=out)


@pytest.fixture
def chapter() -> Chapter:
    return _chapter(
        ([("narration", "Thunder rolled over the point."), ("quote", "Tobias!"), ("narration", "she shouted.")], False),
        ([("quote", "I hear it."), ("narration", "Tobias said.")], False),
        ([("narration", "Morning came grey and quiet, gulls over the rocks.")], True),
        ([("quote", "Well,"), ("narration", "said Hetta."), ("quote", "That is that.")], False),
    )


@pytest.fixture
def bible() -> CastBible:
    return CastBible(characters=[
        CharacterEntry(name="Mara Quill", aliases=["Mara"], line_count=1),
        CharacterEntry(name="Tobias", line_count=1),
        CharacterEntry(name="Hetta", aliases=["Old Hetta"], line_count=2),
    ])


def _ids(script: ChapterScript) -> list[str]:
    return [segment.id for segment in script.segments]


def _assert_contiguous(script: ChapterScript, span_ids: list[str]) -> None:
    index = {span_id: i for i, span_id in enumerate(span_ids)}
    assert script.music, "every chapter gets music"
    assert script.music[0].start_span_id == span_ids[0]
    assert script.music[-1].end_span_id == span_ids[-1]
    for region in script.music:
        assert index[region.start_span_id] <= index[region.end_span_id]
    for previous, current in zip(script.music, script.music[1:]):
        assert index[current.start_span_id] == index[previous.end_span_id] + 1
    assert [region.id for region in script.music] == [f"c{script.chapter_index}m{n:03d}" for n in range(1, len(script.music) + 1)]


# --------------------------------------------------------------------------- segments
def test_segments_correspond_to_spans(chapter: Chapter, bible: CastBible) -> None:
    analyses = [
        ChunkAnalysis(labels=[SpanLabel(span_id="c1p1s1", speaker="Mara", emotion="urgent", delivery="shout"), SpanLabel(span_id="c1p2s0", speaker="Tobias")], source="llm"),
        ChunkAnalysis(labels=[SpanLabel(span_id="c1p4s0", speaker="Old Hetta", emotion="amused"), SpanLabel(span_id="c1p4s2", speaker="Hetta")], source="heuristic"),
    ]
    script = assemble_script(chapter, analyses, bible)
    assert script.chapter_index == 1 and script.title == "Chapter 1"
    assert _ids(script) == [span.id for span in chapter.spans], "one segment per span, in order, no title segment"
    by_id = {segment.id: segment for segment in script.segments}
    assert by_id["c1p1s0"].speaker == NARRATOR and by_id["c1p1s0"].kind == "narration"
    assert by_id["c1p1s1"].speaker == "Mara Quill" and by_id["c1p1s1"].kind == "dialogue"
    assert (by_id["c1p1s1"].emotion, by_id["c1p1s1"].delivery) == ("urgent", "shout")
    assert by_id["c1p4s0"].speaker == "Hetta" and by_id["c1p4s0"].emotion == "amused"
    assert by_id["c1p4s2"].speaker == "Hetta"
    assert by_id["c1p2s0"].text == "I hear it." and by_id["c1p2s0"].paragraph_index == 2
    assert [segment.id for segment in script.segments if segment.scene_break_before] == ["c1p3s0"]
    assert by_id["c1p1s1"].source == "llm" and by_id["c1p2s1"].source == "llm"
    assert by_id["c1p4s0"].source == "heuristic" and by_id["c1p4s1"].source == "heuristic"
    assert script.warnings == []


def test_unknown_speaker_and_missing_label_fall_back_to_narrator(chapter: Chapter, bible: CastBible) -> None:
    analyses = [ChunkAnalysis(labels=[SpanLabel(span_id="c1p1s1", speaker="Somebody Else"), SpanLabel(span_id="c1p2s0", speaker="Tobias")])]
    script = assemble_script(chapter, analyses, bible)
    by_id = {segment.id: segment for segment in script.segments}
    assert by_id["c1p1s1"].speaker == NARRATOR and by_id["c1p1s1"].kind == "dialogue"
    assert by_id["c1p4s0"].speaker == NARRATOR and by_id["c1p4s2"].speaker == NARRATOR
    assert any("Somebody Else" in w for w in script.warnings)
    assert any("c1p4s0" in w for w in script.warnings) and any("c1p4s2" in w for w in script.warnings)


def test_analysis_warnings_are_carried(chapter: Chapter, bible: CastBible) -> None:
    analyses = [ChunkAnalysis(warnings=["chunk said something"])]
    script = assemble_script(chapter, analyses, bible)
    assert "chunk said something" in script.warnings


# --------------------------------------------------------------------------- unknown span ids
def test_unknown_span_ids_are_dropped_with_warnings(chapter: Chapter, bible: CastBible) -> None:
    analyses = [ChunkAnalysis(
        labels=[SpanLabel(span_id="c9p9s9", speaker="Tobias"), SpanLabel(span_id="c1p2s0", speaker="Tobias")],
        sfx_cues=[SfxCueRaw(span_id="c7p7s7", anchor_text="x", description="nothing"), SfxCueRaw(span_id="c1p1s0", anchor_text="Thunder", description="thunder")],
        music_cues=[MusicCueRaw(span_id="c8p8s8", action="start", mood="tense"), MusicCueRaw(span_id="c1p1s0", action="start", mood="calm")],
    )]
    script = assemble_script(chapter, analyses, bible)
    assert _ids(script) == [span.id for span in chapter.spans]
    assert [cue.span_id for cue in script.sfx] == ["c1p1s0"]
    assert [(region.start_span_id, region.mood) for region in script.music] == [("c1p1s0", "calm")]
    for bad in ("c9p9s9", "c7p7s7", "c8p8s8"):
        assert any(bad in w for w in script.warnings)


# --------------------------------------------------------------------------- sfx
def test_sfx_ids_offsets_and_ambient_end_spans(chapter: Chapter, bible: CastBible) -> None:
    analyses = [ChunkAnalysis(sfx_cues=[
        SfxCueRaw(span_id="c1p1s0", anchor_text="thunder", description="thunder", kind="impact", duration_s=2.5, intensity=0.9),
        SfxCueRaw(span_id="c1p1s0", anchor_text="wind", description="wind", kind="ambient", duration_s=15.0),
        SfxCueRaw(span_id="c1p3s0", anchor_text="gulls", description="gulls", kind="ambient", duration_s=12.0),
        SfxCueRaw(span_id="c1p1s2", anchor_text="not there", description="nothing", kind="impact", duration_s=1.0),
    ])]
    script = assemble_script(chapter, analyses, bible)
    assert [cue.id for cue in script.sfx] == ["c1x001", "c1x002", "c1x003", "c1x004"]
    thunder, wind, gulls, missing = script.sfx
    assert thunder.anchor_offset == 0 and thunder.duration_ms == 2500 and thunder.end_span_id is None and thunder.intensity == 0.9
    assert wind.anchor_offset == 0 and wind.end_span_id == "c1p2s1", "ambient runs to the last span before the scene break"
    assert gulls.anchor_offset == chapter.paragraphs[2].spans[0].text.index("gulls") and gulls.end_span_id == "c1p4s2"
    assert missing.anchor_offset == 0
    assert any("not there" in w for w in script.warnings)
    assert not any("thunder" in w for w in script.warnings), "case-insensitive anchor lookup"


# --------------------------------------------------------------------------- music regions
def test_music_actions_resolve_into_regions(chapter: Chapter, bible: CastBible) -> None:
    span_ids = [span.id for span in chapter.spans]                 # 10 spans
    analyses = [
        ChunkAnalysis(music_cues=[MusicCueRaw(span_id="c1p1s0", action="start", mood="tense", energy=0.7, prompt="storm")]),
        ChunkAnalysis(music_cues=[MusicCueRaw(span_id="c1p3s0", action="change", mood="calm", energy=0.3, prompt="")]),
        ChunkAnalysis(music_cues=[MusicCueRaw(span_id="c1p4s2", action="stop")]),
    ]
    script = assemble_script(chapter, analyses, bible)
    assert [(r.id, r.start_span_id, r.end_span_id, r.mood, r.energy) for r in script.music] == [
        ("c1m001", "c1p1s0", "c1p2s1", "tense", 0.7),
        ("c1m002", "c1p3s0", "c1p4s1", "calm", 0.3),
    ]
    assert script.music[0].prompt == "storm" and script.music[1].prompt, "an empty prompt gets the default for its mood"
    assert span_ids[-1] == "c1p4s2"
    assert not any("music" in w for w in script.warnings)


def test_music_region_spans_analysis_boundary(chapter: Chapter, bible: CastBible) -> None:
    analyses = [ChunkAnalysis(music_cues=[MusicCueRaw(span_id="c1p1s0", action="start", mood="warm")]), ChunkAnalysis(), ChunkAnalysis()]
    script = assemble_script(chapter, analyses, bible)
    assert [(r.start_span_id, r.end_span_id, r.mood) for r in script.music] == [("c1p1s0", "c1p4s2", "warm")]


def test_chapter_without_cues_gets_one_region(chapter: Chapter, bible: CastBible) -> None:
    script = assemble_script(chapter, [ChunkAnalysis()], bible)
    assert [(r.id, r.start_span_id, r.end_span_id, r.mood, r.energy) for r in script.music] == [("c1m001", "c1p1s0", "c1p4s2", "calm", 0.2)]
    script = assemble_script(chapter, [ChunkAnalysis()], bible, prior_mood="melancholy")
    assert [(r.mood, r.energy) for r in script.music] == [("melancholy", 0.2)]
    assert script.music[0].prompt


def test_prior_mood_carries_until_first_change(chapter: Chapter, bible: CastBible) -> None:
    analyses = [ChunkAnalysis(music_cues=[MusicCueRaw(span_id="c1p3s0", action="change", mood="calm", energy=0.4)])]
    script = assemble_script(chapter, analyses, bible, prior_mood="tense")
    assert [(r.start_span_id, r.end_span_id, r.mood) for r in script.music] == [("c1p1s0", "c1p2s1", "tense"), ("c1p3s0", "c1p4s2", "calm")]
    same_span = [ChunkAnalysis(music_cues=[MusicCueRaw(span_id="c1p1s0", action="change", mood="calm", energy=0.4)])]
    script = assemble_script(chapter, same_span, bible, prior_mood="tense")
    assert [(r.start_span_id, r.end_span_id, r.mood, r.energy) for r in script.music] == [("c1p1s0", "c1p4s2", "calm", 0.4)]


def test_stop_without_region_warns_and_no_cues_without_prior_leaves_gap(chapter: Chapter, bible: CastBible) -> None:
    analyses = [ChunkAnalysis(music_cues=[MusicCueRaw(span_id="c1p2s0", action="stop"), MusicCueRaw(span_id="c1p3s0", action="start", mood="hopeful")])]
    script = assemble_script(chapter, analyses, bible)
    assert [(r.start_span_id, r.end_span_id, r.mood) for r in script.music] == [("c1p3s0", "c1p4s2", "hopeful")]
    assert any("stop" in w for w in script.warnings)


def test_empty_chapter(bible: CastBible) -> None:
    script = assemble_script(Chapter(index=4, title="Nothing", paragraphs=[]), [], bible)
    assert script.segments == [] and script.music == [] and script.sfx == []


# --------------------------------------------------------------------------- the fixture through real chunks
def _run_book(book, max_chars: int):
    analyzer = HeuristicAnalyzer()
    bible = CastBible()
    prior = "none"
    per_chapter = {}
    for chapter in book.chapters:
        analyses, chunks, chapter_prior = [], [], prior
        for chunk in make_chunks(chapter, max_chars):
            chunk = chunk.model_copy(update={"prior_mood": prior})
            analysis, _ = validate_chunk_analysis(analyzer.analyze_chunk(chunk, bible), chunk, bible)
            bible = apply_updates(bible, analysis.characters, chapter.index)
            for label in analysis.labels:
                bible, _ = register_speaker(bible, label.speaker, chapter.index)
            for cue in analysis.music_cues:
                prior = "none" if cue.action == "stop" else cue.mood
            analyses.append(analysis)
            chunks.append(chunk)
        per_chapter[chapter.index] = (chapter, analyses, chunks, chapter_prior)
    bible = finalize(bible)
    return {index: assemble_script(chapter, analyses, bible, prior_mood=chapter_prior) for index, (chapter, analyses, _, chapter_prior) in per_chapter.items()}, per_chapter


@pytest.fixture(scope="module")
def book(sample_book_path: Path):
    return load_book(sample_book_path)


def test_scene_spanning_chunk_boundary_yields_one_region(book) -> None:
    scripts, per_chapter = _run_book(book, max_chars=300)
    chapter, _, chunks, _ = per_chapter[2]
    assert len(chunks) >= 4, "chapter 2 must be split into several chunks"
    span_ids = [span.id for span in chapter.spans]
    script = scripts[2]
    assert [(r.start_span_id, r.end_span_id, r.mood) for r in script.music] == [("c2p1s0", "c2p8s1", "calm"), ("c2p9s0", span_ids[-1], "warm")]
    boundaries = {chunk.spans[0].id for chunk in chunks[1:]}
    calm_span_ids = span_ids[: span_ids.index("c2p8s1") + 1]
    assert len(boundaries & set(calm_span_ids)) >= 2, "the calm scene crosses chunk boundaries and stays one region"
    _assert_contiguous(script, span_ids)


def test_fixture_regions_are_contiguous_and_ambient_cues_end_at_chapter_end(book) -> None:
    scripts, per_chapter = _run_book(book, max_chars=6000)
    for index, script in scripts.items():
        chapter = per_chapter[index][0]
        span_ids = [span.id for span in chapter.spans]
        assert _ids(script) == span_ids
        _assert_contiguous(script, span_ids)
        for cue in script.sfx:
            segment = next(segment for segment in script.segments if segment.id == cue.span_id)
            assert segment.text.lower().index(cue.anchor_text.lower()) == cue.anchor_offset
            assert cue.end_span_id == (span_ids[-1] if cue.kind == "ambient" else None), "the fixture has no scene breaks"
        assert [cue.id for cue in script.sfx] == [f"c{index}x{n:03d}" for n in range(1, len(script.sfx) + 1)]
    assert [r.mood for r in scripts[1].music] == ["tense"]
    assert [r.mood for r in scripts[2].music] == ["calm", "warm"]
    assert [r.mood for r in scripts[3].music] == ["hopeful", "melancholy"]
    assert scripts[2].music[0].start_span_id == "c2p1s0", "chapter 2 opens with its own change, not a carried tense region"
    speakers = {segment.speaker for script in scripts.values() for segment in script.segments if segment.kind == "dialogue"}
    assert speakers == {"Mara Quill", "Tobias", "Ansel Vey", "Hetta"}
    assert all(script.warnings == [] for script in scripts.values())
