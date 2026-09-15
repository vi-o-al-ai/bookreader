"""HeuristicAnalyzer on the fixture: the attribution table, emotions, sfx, music, the final
bible, determinism, graceful degradation; plus validate_chunk_analysis repairs."""
from __future__ import annotations

from pathlib import Path

import pytest

from bookreader.analysis.bible import apply_updates, finalize, register_speaker
from bookreader.analysis.chunker import carry_speakers, make_chunks
from bookreader.analysis.validate import AnalysisInvalid, validate_chunk_analysis
from bookreader.ingest import load_book
from bookreader.providers.base import NullUsage, TextAnalyzer, resolve
from bookreader.providers.mock.analysis import HeuristicAnalyzer
from bookreader.types import NARRATOR, Book, CastBible, CharacterUpdate, Chunk, ChunkAnalysis, MusicCueRaw, SfxCueRaw, Span, SpanLabel

# span id -> (quote text prefix, expected speaker)
EXPECTED_SPEAKERS: dict[str, tuple[str, str]] = {
    "c1p2s0": ("Tobias!", "Mara Quill"),
    "c1p2s2": ("The bell! Ring the bell!", "Mara Quill"),
    "c1p3s1": ("I can't reach the rope", "Tobias"),
    "c1p6s0": ("Stand back,", "Mara Quill"),
    "c1p6s2": ("When this comes free", "Mara Quill"),
    "c1p8s0": ("Do you think they heard?", "Tobias"),
    "c1p9s0": ("They heard,", "Mara Quill"),
    "c2p3s0": ("Easy,", "Mara Quill"),
    "c2p3s2": ("You're on Gull Point.", "Mara Quill"),
    "c2p4s1": ("The Corvid,", "Ansel Vey"),
    "c2p4s3": ("My ship. Did anyone else...", "Ansel Vey"),
    "c2p5s0": ("Only you, so far,", "Mara Quill"),
    "c2p5s2": ("What's your name?", "Mara Quill"),
    "c2p6s0": ("Ansel. Ansel Vey.", "Ansel Vey"),
    "c2p6s2": ("First mate.", "Ansel Vey"),
    "c2p7s1": ("Is he a pirate?", "Tobias"),
    "c2p8s1": ("Not today, lad.", "Ansel Vey"),
    "c2p10s0": ("You rang the bell,", "Ansel Vey"),
    "c2p11s0": ("We did.", "Mara Quill"),
    "c2p12s0": ("Then you saved my life.", "Ansel Vey"),
    "c2p12s2": ("I'll not forget it.", "Ansel Vey"),
    "c2p13s1": ("Fine words,", "Hetta"),
    "c2p13s3": ("Words don't mend a roof.", "Hetta"),
    "c2p14s0": ("Hetta,", "Mara Quill"),
    "c2p15s0": ("I'm only saying.", "Hetta"),
    "c3p4s0": ("What is it?", "Ansel Vey"),
    "c3p5s0": ("The Harbour Board,", "Mara Quill"),
    "c3p5s2": ("They're closing the light.", "Mara Quill"),
    "c3p6s1": ("Leave? But where would we go?", "Tobias"),
    "c3p8s1": ("Well,", "Hetta"),
    "c3p8s3": ("I suppose someone had better", "Hetta"),
}

EXPECTED_SFX_ANCHORS = ("thunder", "slammed", "snap", "bell", "gulls", "crunching", "crackled", "whistle", "hooves", "needles clicking")
EXPECTED_MUSIC = {1: [("start", "tense")], 2: [("change", "calm"), ("change", "warm")], 3: [("change", "hopeful"), ("change", "melancholy")]}


class BookRun:
    """The analyze stage in miniature: chunks in order, bible threaded, prior_mood and prior_speakers carried."""

    def __init__(self, book: Book, max_chars: int = 6000) -> None:
        self.book = book
        self.analyses: dict[int, list[ChunkAnalysis]] = {}
        self.chunks: dict[int, list[Chunk]] = {}
        self.chapter_prior: dict[int, str] = {}
        self.labels: dict[str, SpanLabel] = {}
        self.spans = {span.id: span for chapter in book.chapters for span in chapter.spans}
        analyzer = HeuristicAnalyzer()
        bible = CastBible()
        prior = "none"
        for chapter in book.chapters:
            self.chapter_prior[chapter.index] = prior
            self.analyses[chapter.index] = []
            self.chunks[chapter.index] = []
            speakers: list[str] = []
            for chunk in make_chunks(chapter, max_chars):
                chunk = chunk.model_copy(update={"prior_mood": prior, "prior_speakers": speakers})
                analysis = analyzer.analyze_chunk(chunk, bible)
                analysis, _ = validate_chunk_analysis(analysis, chunk, bible)
                bible = apply_updates(bible, analysis.characters, chapter.index)
                for label in analysis.labels:
                    bible, _ = register_speaker(bible, label.speaker, chapter.index)
                    self.labels[label.span_id] = label
                for cue in analysis.music_cues:
                    prior = "none" if cue.action == "stop" else cue.mood
                speakers = carry_speakers(speakers, chunk, analysis)
                self.analyses[chapter.index].append(analysis)
                self.chunks[chapter.index].append(chunk)
        self.raw_bible = bible
        self.bible = finalize(bible)

    def cues(self, chapter_index: int) -> list[SfxCueRaw]:
        return [cue for analysis in self.analyses[chapter_index] for cue in analysis.sfx_cues]

    def music(self, chapter_index: int) -> list[MusicCueRaw]:
        return [cue for analysis in self.analyses[chapter_index] for cue in analysis.music_cues]


@pytest.fixture(scope="module")
def book(sample_book_path: Path) -> Book:
    return load_book(sample_book_path)


@pytest.fixture(scope="module")
def run(book: Book) -> BookRun:
    return BookRun(book)


# --------------------------------------------------------------------------- attribution
def test_every_quote_span_is_labelled_once(run: BookRun, book: Book) -> None:
    quote_ids = [span.id for chapter in book.chapters for span in chapter.spans if span.kind == "quote"]
    assert sorted(run.labels) == sorted(quote_ids)
    assert set(EXPECTED_SPEAKERS) == set(quote_ids), "the expected table covers every quote in the fixture"
    for analyses in run.analyses.values():
        ids = [label.span_id for analysis in analyses for label in analysis.labels]
        assert len(ids) == len(set(ids))


@pytest.mark.parametrize("span_id", list(EXPECTED_SPEAKERS))
def test_attribution_table(run: BookRun, span_id: str) -> None:
    prefix, speaker = EXPECTED_SPEAKERS[span_id]
    assert run.spans[span_id].text.startswith(prefix), "the table matches the fixture text"
    assert run.labels[span_id].speaker == speaker


def test_emotions_and_deliveries(run: BookRun) -> None:
    assert (run.labels["c1p2s0"].emotion, run.labels["c1p2s0"].delivery) == ("urgent", "shout")
    assert (run.labels["c1p2s2"].emotion, run.labels["c1p2s2"].delivery) == ("urgent", "shout")
    assert run.labels["c1p3s1"].emotion == "afraid"
    assert (run.labels["c2p4s1"].emotion, run.labels["c2p4s1"].delivery) == ("afraid", "whisper")
    assert run.labels["c2p4s3"].delivery == "whisper"
    assert (run.labels["c1p6s0"].emotion, run.labels["c1p6s0"].delivery) == ("calm", "quiet")
    assert run.labels["c1p6s2"].delivery == "quiet"
    assert run.labels["c2p8s1"].emotion == "amused"
    assert run.labels["c2p14s0"].emotion == "stern"
    assert run.labels["c2p12s2"].emotion == "tender"
    assert run.labels["c2p7s1"].emotion == "hesitant", "a question from a child"
    assert (run.labels["c2p5s2"].emotion, run.labels["c2p5s2"].delivery) == ("neutral", "normal")
    assert run.labels["c2p11s0"].delivery == "normal"


# --------------------------------------------------------------------------- sfx
def test_sfx_anchors_are_verbatim_and_on_narration(run: BookRun) -> None:
    cues = [cue for chapter in (1, 2, 3) for cue in run.cues(chapter)]
    anchors = {cue.anchor_text.lower() for cue in cues}
    for expected in EXPECTED_SFX_ANCHORS:
        assert expected in anchors
    for cue in cues:
        span = run.spans[cue.span_id]
        assert span.kind == "narration"
        assert cue.anchor_text in span.text
        assert 0.5 <= cue.duration_s <= 30 and 0 <= cue.intensity <= 1


def test_sfx_details(run: BookRun) -> None:
    by_anchor = {cue.anchor_text.lower(): cue for chapter in (1, 2, 3) for cue in run.cues(chapter)}
    assert by_anchor["thunder"].kind == "impact" and by_anchor["thunder"].span_id == "c1p4s0"
    assert by_anchor["slammed"].description == "shutter banging repeatedly" and by_anchor["slammed"].duration_s == 4.0
    assert by_anchor["bell"].span_id == "c1p7s0" and "two bell strikes" in by_anchor["bell"].description
    assert by_anchor["gulls"].kind == "ambient" and by_anchor["gulls"].span_id == "c2p1s0"
    assert by_anchor["crackled"].kind == "ambient" and by_anchor["whistle"].kind == "impact"
    assert by_anchor["whistle"].span_id == "c2p9s0" and by_anchor["crackled"].span_id == "c2p9s0"
    assert by_anchor["hooves"].span_id == "c3p2s0"
    assert by_anchor["needles clicking"].kind == "ambient"
    assert by_anchor["whistling"].description == "a man whistling a tune while he works"
    per_paragraph: dict[str, int] = {}
    for chapter in (1, 2, 3):
        for cue in run.cues(chapter):
            key = cue.span_id.rsplit("s", 1)[0]
            per_paragraph[key] = per_paragraph.get(key, 0) + 1
    assert max(per_paragraph.values()) <= 3


# --------------------------------------------------------------------------- music
def test_music_sequence_per_chapter(run: BookRun) -> None:
    for chapter, expected in EXPECTED_MUSIC.items():
        assert [(cue.action, cue.mood) for cue in run.music(chapter)] == expected
    assert run.music(1)[0].span_id == "c1p1s0" and run.music(1)[0].energy == 0.7
    assert run.music(2)[0].span_id == "c2p1s0" and run.music(2)[1].span_id == "c2p9s0"
    assert run.music(3)[0].span_id == "c3p1s0" and run.music(3)[1].span_id == "c3p2s0"
    assert all(cue.prompt for chapter in (1, 2, 3) for cue in run.music(chapter))


def test_music_starts_when_nothing_is_playing(book: Book) -> None:
    chunk = make_chunks(book.chapters[1], 6000)[0]
    analysis = HeuristicAnalyzer().analyze_chunk(chunk, CastBible())
    assert [(cue.action, cue.mood) for cue in analysis.music_cues] == [("start", "calm"), ("change", "warm")]


# --------------------------------------------------------------------------- bible
def test_final_bible_has_exactly_the_four_characters(run: BookRun) -> None:
    entries = {entry.name: entry for entry in run.bible.characters}
    assert set(entries) == {"Mara Quill", "Tobias", "Ansel Vey", "Hetta"}
    assert (entries["Mara Quill"].gender, entries["Mara Quill"].age, set(entries["Mara Quill"].aliases)) == ("female", "adult", {"Mara"})
    assert (entries["Tobias"].gender, entries["Tobias"].age) == ("male", "child")
    assert (entries["Ansel Vey"].gender, entries["Ansel Vey"].age) == ("male", "adult")
    assert set(entries["Ansel Vey"].aliases) == {"Ansel", "the stranger", "the man"}
    assert (entries["Hetta"].gender, entries["Hetta"].age, set(entries["Hetta"].aliases)) == ("female", "elderly", {"Old Hetta"})
    assert not any(entry.provisional for entry in run.bible.characters)
    assert {name: entry.line_count for name, entry in entries.items()} == {"Mara Quill": 13, "Tobias": 4, "Ansel Vey": 9, "Hetta": 5}
    assert {name: entry.first_chapter for name, entry in entries.items()} == {"Mara Quill": 1, "Tobias": 1, "Ansel Vey": 2, "Hetta": 2}
    assert entries["Tobias"].description.startswith("Tobias was only twelve")
    assert entries["Ansel Vey"].voice_notes == "dry, rasping"


def test_bible_is_built_incrementally(run: BookRun) -> None:
    analysis_ch1 = run.analyses[1][0]
    assert {u.name for u in analysis_ch1.characters} == {"Mara Quill", "Tobias"}
    ansel = next(u for u in run.analyses[2][0].characters if u.name == "Ansel Vey")
    assert ansel.merge_into is None and set(ansel.aliases) == {"Ansel", "the stranger", "the man"}


# --------------------------------------------------------------------------- determinism and degradation
def test_two_runs_are_identical(book: Book, run: BookRun) -> None:
    again = BookRun(book)
    for chapter in (1, 2, 3):
        assert [a.model_dump() for a in again.analyses[chapter]] == [a.model_dump() for a in run.analyses[chapter]]
    assert again.bible == run.bible
    assert again.bible.fingerprint() == run.bible.fingerprint()


def test_no_warnings_on_the_fixture(run: BookRun) -> None:
    assert [w for chapter in (1, 2, 3) for a in run.analyses[chapter] for w in a.warnings] == []


def test_small_chunks_still_resolve_name_introduction(book: Book) -> None:
    small = BookRun(book, max_chars=300)
    assert len(small.chunks[2]) >= 4
    assert small.labels["c2p4s1"].speaker == "the stranger", "the descriptor is the best name available in its chunk"
    assert small.labels["c2p6s0"].speaker == "Ansel Vey"
    introduced = next(u for a in small.analyses[2] for u in a.characters if u.name == "Ansel Vey")
    assert introduced.merge_into == "the stranger"
    ansel = small.bible.find("the stranger")
    assert ansel is not None and ansel.name == "Ansel Vey" and not ansel.provisional
    assert {entry.name for entry in small.bible.characters} == {"Mara Quill", "Tobias", "Ansel Vey", "Hetta"}
    assert [(c.action, c.mood) for c in small.music(2)] == [("change", "calm"), ("change", "warm")]


def test_small_chunks_keep_the_conversation_across_chunk_boundaries(book: Book, run: BookRun) -> None:
    small = BookRun(book, max_chars=300)
    narrated = [span_id for span_id, label in small.labels.items() if label.speaker == NARRATOR]
    assert narrated == [], "no quote falls to the narrator just because a chunk cut landed mid-exchange"
    assert small.labels["c2p12s0"].speaker == small.labels["c2p12s2"].speaker == "Ansel Vey"
    assert small.labels["c2p12s0"].speaker == run.labels["c2p12s0"].speaker, "same answer as at 6000 chars"
    assert small.chunks[2][-1].prior_speakers, "the stage passes who spoke last into the next chunk"


def test_untagged_exchange_alternates_across_a_chunk_cut(tmp_path: Path) -> None:
    lines = ["I never said that.", "You did, and you know it.", "Then prove it.", "I don't have to.", "Fine.", "Fine."]
    text = (
        "Ilse found her brother Aldo in the barn, sitting on a bucket with his head in his hands, and the "
        "argument that had been waiting all week began before either of them had thought of a kinder way in.\n\n"
        '"You sold the mare," Ilse said.\n\n"I had no choice," said Aldo.\n\n' + "\n\n".join(f'"{line}"' for line in lines) + "\n"
    )
    path = tmp_path / "cut.txt"
    path.write_text(text, encoding="utf-8")
    chapter = load_book(path).chapters[0]
    chunks = make_chunks(chapter, 120)
    assert len(chunks) >= 3 and all(not any(NAME in c.context_before for NAME in ("Ilse", "Aldo")) for c in chunks[2:]), \
        "later chunks see only bare quotes in their context"
    analyzer, bible, speakers, labels = HeuristicAnalyzer(), CastBible(), [], {}
    for chunk in chunks:
        chunk = chunk.model_copy(update={"prior_speakers": speakers})
        analysis = analyzer.analyze_chunk(chunk, bible)
        bible = apply_updates(bible, analysis.characters, 1)
        for label in analysis.labels:
            bible, _ = register_speaker(bible, label.speaker, 1)
            labels[label.span_id] = label.speaker
        speakers = carry_speakers(speakers, chunk, analysis)
    ordered = [labels[f"c1p{n}s0"] for n in range(2, 10)]
    assert ordered == ["Ilse", "Aldo"] * 4, ordered

    stateless = HeuristicAnalyzer().analyze_chunk(chunks[-1], bible)
    assert any(label.speaker == NARRATOR for label in stateless.labels), "without prior_speakers the tail chunk falls to the narrator"


def test_three_speaker_untagged_exchange_degrades_gracefully(tmp_path: Path) -> None:
    text = (
        "Ann looked at Bob and then at Cal. The three of them stood on the quay while the tide came in and "
        "the boats knocked together below them in the grey light of the early morning, none of them willing "
        "to be the first to speak about the letter that had come up from the town.\n\n"
        '"Shall we go?" Ann asked.\n\n"Not yet," Bob said.\n\n"Why not?" said Cal.\n\n'
        '"Because it is raining."\n\n"So what?"\n\n"Fine."\n\n"Then we stay."\n'
    )
    path = tmp_path / "three.txt"
    path.write_text(text, encoding="utf-8")
    chapter = load_book(path).chapters[0]
    chunk = make_chunks(chapter, 6000)[0]
    analysis = HeuristicAnalyzer().analyze_chunk(chunk, CastBible())
    quote_ids = [span.id for span in chunk.spans if span.kind == "quote"]
    labelled = {label.span_id: label.speaker for label in analysis.labels}
    assert sorted(labelled) == sorted(quote_ids)
    assert labelled["c1p2s0"] == "Ann" and labelled["c1p3s0"] == "Bob" and labelled["c1p4s0"] == "Cal"
    assert set(labelled.values()) <= {"Ann", "Bob", "Cal"}
    assert analysis.warnings, "guesses among three participants are reported"
    assert any("ambiguous" in warning for warning in analysis.warnings)
    assert {u.name for u in analysis.characters} == {"Ann", "Bob", "Cal"}


@pytest.mark.parametrize("exclamation", ["Riders,", "Nonsense,", "Careful!", "Quiet!", "Liar!"])
def test_one_word_exclamations_are_not_vocatives_and_create_no_character(tmp_path: Path, exclamation: str) -> None:
    text = (
        "Halloran took the first watch while Brannock slept, and the fire had burned down to embers by the time "
        "the wind moved in the dry grass beyond the camp and the first sound of the road reached them.\n\n"
        '"Wake me at midnight," said Brannock.\n\n"I will," Halloran said.\n\n'
        f'"{exclamation}" Halloran whispered.\n\n'
        'Brannock was awake at once. "How many?"\n\n"Two. Maybe three."\n'
    )
    path = tmp_path / "riders.txt"
    path.write_text(text, encoding="utf-8")
    chapter = load_book(path).chapters[0]
    chunk = make_chunks(chapter, 6000)[0]
    analysis = HeuristicAnalyzer().analyze_chunk(chunk, CastBible())
    labelled = {label.span_id: label.speaker for label in analysis.labels}
    assert labelled["c1p4s0"] == "Halloran"
    assert labelled["c1p6s0"] == "Halloran", "the reply to Brannock's question is not given to a phantom addressee"
    assert {u.name for u in analysis.characters} == {"Halloran", "Brannock"}


def test_vocative_naming_a_character_seen_in_narration_is_the_addressee(book: Book, run: BookRun) -> None:
    # 'Tobias!' opens chapter 1 before any tag names him; narration ('Tobias was only twelve') corroborates the name
    assert run.labels["c1p2s0"].speaker == "Mara Quill" and run.labels["c1p3s1"].speaker == "Tobias"


def test_unattributable_quote_goes_to_narrator_with_warning() -> None:
    chunk = Chunk(
        chapter_index=1, chunk_index=0, paragraph_start=1, paragraph_end=1,
        spans=[Span(id="c1p1s0", kind="quote", text="Nobody knows.", start_char=1, end_char=14)],
    )
    analysis = HeuristicAnalyzer().analyze_chunk(chunk, CastBible())
    assert analysis.labels == [SpanLabel(span_id="c1p1s0", speaker=NARRATOR)]
    assert analysis.warnings and "c1p1s0" in analysis.warnings[0]
    assert analysis.characters == []


def test_empty_chunk() -> None:
    chunk = Chunk(chapter_index=1, chunk_index=0, paragraph_start=1, paragraph_end=1, spans=[])
    analysis = HeuristicAnalyzer().analyze_chunk(chunk, CastBible())
    assert analysis == ChunkAnalysis(source="heuristic")


# --------------------------------------------------------------------------- scene breaks
def test_scene_break_resets_the_conversation(tmp_path: Path) -> None:
    # Before the break Ann and Bob alternate and Bob addresses Ann by name, so the fallback rule
    # has a certain partner in hand. After the break narration introduces Cal and an untagged
    # quote follows: with the conversation reset the only participant is Cal.
    text = (
        "Ann and Bob stood on the quay in the grey light while the tide came in and the boats knocked together "
        "below them, neither of them willing to be the first to speak about the letter that had come up from the "
        "town that morning.\n\n"
        '"Shall we go?" Ann asked.\n\n'
        'Bob shook his head. "Not yet, Ann."\n\n'
        '"Why not?"\n\n'
        '"Because it is raining."\n\n\n\n'
        "Cal came down the cliff path with a lantern in his hand. Cal stopped at the edge of the water and "
        "lifted the light.\n\n"
        '"Is anyone there?"\n'
    )
    path = tmp_path / "scene.txt"
    path.write_text(text, encoding="utf-8")
    chapter = load_book(path).chapters[0]
    assert [p.index for p in chapter.paragraphs if p.scene_break_before] == [6]
    chunk = make_chunks(chapter, 6000)[0]
    assert chunk.scene_break_paragraphs == [6]

    analysis = HeuristicAnalyzer().analyze_chunk(chunk, CastBible())
    labelled = {label.span_id: label.speaker for label in analysis.labels}
    assert sorted(labelled) == sorted(span.id for span in chunk.spans if span.kind == "quote")
    assert [labelled[i] for i in ("c1p2s0", "c1p3s1", "c1p4s0", "c1p5s0")] == ["Ann", "Bob", "Ann", "Bob"]
    assert labelled["c1p7s0"] == "Cal", "after the break the newly introduced character speaks"
    assert analysis.warnings == []
    assert {u.name for u in analysis.characters} == {"Ann", "Bob", "Cal"}

    unaware = HeuristicAnalyzer().analyze_chunk(chunk.model_copy(update={"scene_break_paragraphs": []}), CastBible())
    assert any("ambiguous" in w and "Ann" in w for w in unaware.warnings), \
        "without the scene-break info the pre-break participants stay candidates for the new scene's first quote"


def test_scene_break_lets_the_mood_change_without_lookahead() -> None:
    def paragraph(index: int, text: str) -> Span:
        return Span(id=f"c1p{index}s0", kind="narration", text=text, start_char=0, end_char=len(text))

    spans = [paragraph(1, "The morning was quiet."), paragraph(2, "A storm broke over the roof."), paragraph(3, "Nothing else happened.")]
    with_break = Chunk(chapter_index=1, chunk_index=0, paragraph_start=1, paragraph_end=3, spans=spans, scene_break_paragraphs=[2])
    cues = HeuristicAnalyzer().analyze_chunk(with_break, CastBible()).music_cues
    assert [(cue.span_id, cue.action, cue.mood) for cue in cues] == [("c1p1s0", "start", "calm"), ("c1p2s0", "change", "tense")]

    without = with_break.model_copy(update={"scene_break_paragraphs": []})
    cues = HeuristicAnalyzer().analyze_chunk(without, CastBible()).music_cues
    assert [(cue.action, cue.mood) for cue in cues] == [("start", "calm")], "an unconfirmed one-paragraph mood is otherwise ignored"


# --------------------------------------------------------------------------- provider contract
def test_provider_contract(mock_settings) -> None:
    analyzer = HeuristicAnalyzer()
    assert isinstance(analyzer, TextAnalyzer)
    assert (HeuristicAnalyzer.family, analyzer.cache_version, analyzer.model_id) == ("mock", "2", "heuristic-1")
    assert HeuristicAnalyzer.check(mock_settings) == []
    built = HeuristicAnalyzer.from_settings(mock_settings, NullUsage())
    assert isinstance(built, HeuristicAnalyzer)
    assert built.warmup() is None
    assert resolve("analysis", "mock") is HeuristicAnalyzer


def test_label_missing_returns_only_requested_ids(book: Book) -> None:
    chunk = make_chunks(book.chapters[0], 6000)[0]
    labels = HeuristicAnalyzer().label_missing(chunk, CastBible(), ["c1p3s1", "c1p9s0"])
    assert [(label.span_id, label.speaker) for label in labels] == [("c1p3s1", "Tobias"), ("c1p9s0", "Mara Quill")]


# --------------------------------------------------------------------------- validate_chunk_analysis
@pytest.fixture
def ch1_chunk(book: Book) -> Chunk:
    return make_chunks(book.chapters[0], 6000)[0]


def _labels_for(chunk: Chunk, speaker: str = "Mara Quill") -> list[SpanLabel]:
    return [SpanLabel(span_id=span.id, speaker=speaker) for span in chunk.spans if span.kind == "quote"]


def test_validate_passes_clean_heuristic_output(ch1_chunk: Chunk) -> None:
    analysis = HeuristicAnalyzer().analyze_chunk(ch1_chunk, CastBible())
    repaired, warnings = validate_chunk_analysis(analysis, ch1_chunk, CastBible())
    assert warnings == [] and repaired == analysis


def test_validate_drops_unknown_labels_and_fills_missing_with_heuristic(ch1_chunk: Chunk) -> None:
    labels = _labels_for(ch1_chunk)
    labels = [label for label in labels if label.span_id != "c1p3s1"]          # 1 of 7 missing (< 20 %)
    labels.append(SpanLabel(span_id="c9p9s9", speaker="Nobody"))
    labels.append(SpanLabel(span_id="c1p2s0", speaker="Tobias"))                 # duplicate, first wins
    analysis = ChunkAnalysis(labels=labels, source="llm")
    repaired, warnings = validate_chunk_analysis(analysis, ch1_chunk, CastBible())
    by_id = {label.span_id: label for label in repaired.labels}
    assert "c9p9s9" not in by_id
    assert by_id["c1p3s1"].speaker == "Tobias" and by_id["c1p3s1"].emotion == "afraid"
    assert by_id["c1p2s0"].speaker == "Mara Quill"
    assert [label.span_id for label in repaired.labels] == [span.id for span in ch1_chunk.spans if span.kind == "quote"]
    assert any("c9p9s9" in w for w in warnings) and any("c1p3s1" in w for w in warnings) and any("duplicate" in w for w in warnings)
    assert repaired.warnings == warnings and repaired.source == "llm"


def test_validate_drops_labels_on_narration_spans(ch1_chunk: Chunk, book: Book) -> None:
    narration = next(span for span in ch1_chunk.spans if span.kind == "narration")
    labels = [*_labels_for(ch1_chunk), SpanLabel(span_id=narration.id, speaker="Tobias", emotion="angry", delivery="shout")]
    repaired, warnings = validate_chunk_analysis(ChunkAnalysis(labels=labels, source="llm"), ch1_chunk, CastBible())
    assert narration.id not in {label.span_id for label in repaired.labels}
    assert any(narration.id in w and "narration" in w for w in warnings)
    from bookreader.analysis.assemble import assemble_script

    unrepaired = ChunkAnalysis(labels=labels, source="llm")
    segment = next(s for s in assemble_script(book.chapters[0], [unrepaired], CastBible(), "none").segments if s.id == narration.id)
    assert (segment.speaker, segment.emotion, segment.delivery) == (NARRATOR, "neutral", "normal"), "assembly never lets a label shout the narrator"


def test_validate_resolves_specific_descriptor_speakers_to_the_known_character(ch1_chunk: Chunk) -> None:
    bible = apply_updates(CastBible(), [CharacterUpdate(name="Mara Quill", gender="female", age="adult"), CharacterUpdate(name="Tobias", gender="male", age="child")], 1)
    repaired, warnings = validate_chunk_analysis(ChunkAnalysis(labels=_labels_for(ch1_chunk, speaker="the boy"), source="llm"), ch1_chunk, bible)
    assert {label.speaker for label in repaired.labels} == {"Tobias"}
    for label in repaired.labels:
        bible, canonical = register_speaker(bible, label.speaker, 1)
        assert canonical == "Tobias"
    assert [entry.name for entry in finalize(bible).characters] == ["Mara Quill", "Tobias"], "no separate 'The Boy'"


def test_validate_rejects_too_many_missing_labels(ch1_chunk: Chunk) -> None:
    analysis = ChunkAnalysis(labels=_labels_for(ch1_chunk)[:4])                 # 3 of 7 missing
    with pytest.raises(AnalysisInvalid) as info:
        validate_chunk_analysis(analysis, ch1_chunk, CastBible())
    assert "c1p8s0" in str(info.value) and "c1p9s0" in str(info.value)


def test_validate_rejects_cues_outside_the_chunk(ch1_chunk: Chunk) -> None:
    analysis = ChunkAnalysis(
        labels=_labels_for(ch1_chunk),
        sfx_cues=[SfxCueRaw(span_id="c2p1s0", anchor_text="Gulls", description="gulls")],
        music_cues=[MusicCueRaw(span_id="c3p1s0", action="start", mood="calm")],
    )
    with pytest.raises(AnalysisInvalid) as info:
        validate_chunk_analysis(analysis, ch1_chunk, CastBible())
    assert "c2p1s0" in str(info.value) and "c3p1s0" in str(info.value)


def test_validate_coerces_speakers_and_clamps_cues(ch1_chunk: Chunk) -> None:
    bible = apply_updates(CastBible(), [CharacterUpdate(name="Mara Quill", aliases=["Mara"])], 1)
    labels = _labels_for(ch1_chunk, speaker="mara")
    labels[0] = labels[0].model_copy(update={"speaker": "the boy"})
    labels[1] = labels[1].model_copy(update={"speaker": "narrator"})
    analysis = ChunkAnalysis(
        labels=labels,
        characters=[CharacterUpdate(name="Tobias", aliases=["the boy"], gender="male", age="child")],
        sfx_cues=[SfxCueRaw(span_id="c1p4s0", anchor_text="Thunder", description="thunder", duration_s=90.0, intensity=1.7)],
        music_cues=[MusicCueRaw(span_id="c1p1s0", action="start", mood="tense", energy=-0.5)],
    )
    repaired, warnings = validate_chunk_analysis(analysis, ch1_chunk, bible)
    speakers = [label.speaker for label in repaired.labels]
    assert speakers[0] == "Tobias" and speakers[1] == NARRATOR and set(speakers[2:]) == {"Mara Quill"}
    assert repaired.sfx_cues[0].duration_s == 30.0 and repaired.sfx_cues[0].intensity == 1.0
    assert repaired.music_cues[0].energy == 0.0
    assert any("clamped" in w for w in warnings) and any("coerced" in w for w in warnings)
    assert analysis.sfx_cues[0].duration_s == 90.0, "input untouched"


# --------------------------------------------------------------------------- attribution heuristics (regressions)
ARGUMENT = (
    "Rain drummed on the kitchen window while Dorian Ash set three cups on the table. His sister Petra had not sat "
    "down, and their cousin Silas was leaning against the door frame with his arms folded.\n\n"
    '"You sold it," Petra said. "You sold Father\'s boat without asking either of us."\n\n'
    '"I asked," Dorian said. "I asked in March. You said do what you like."\n\n'
    '"I said no such thing, Dorian."\n\n"You did, Petra. Silas was there."\n\n"Leave me out of it."\n\n'
    '"You can\'t be left out of it, Silas. It was your name on the paper too."\n\n'
    '"Then it was my name on a paper I never read."\n\n"Oh, that is convenient."\n\n"It\'s not convenient, it\'s true!"\n'
)


def _run_text(tmp_path: Path, name: str, text: str, max_chars: int = 6000) -> BookRun:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return BookRun(load_book(path), max_chars=max_chars)


def test_mid_quote_vocative_and_a_quote_is_never_spoken_by_its_addressee(tmp_path: Path) -> None:
    run = _run_text(tmp_path, "argument.txt", ARGUMENT)
    speakers = {span_id: label.speaker for span_id, label in run.labels.items()}
    assert speakers["c1p5s0"] == "Dorian Ash", "'You did, Petra.' addresses Petra, so Petra does not speak it"
    assert speakers["c1p7s0"] == "Dorian Ash", "a vocative before an interior sentence break is still a vocative"
    assert speakers["c1p8s0"] == "Silas" and speakers["c1p10s0"] == "Silas"
    silas = run.bible.find("Silas")
    assert silas is not None and silas.line_count > 0, "the third speaker is cast"


def test_gender_from_kinship_words_and_a_name_bounded_pronoun_window(tmp_path: Path) -> None:
    run = _run_text(tmp_path, "argument.txt", ARGUMENT)
    genders = {entry.name: entry.gender for entry in run.bible.characters}
    assert genders == {"Dorian Ash": "male", "Petra": "female", "Silas": "male"}, "'His sister Petra' is not made male by Silas's 'his'"
    text = (
        "Marie waited on the bridge until the bells had stopped. Tomas came along the towpath with his coat over his "
        "arm, in no hurry at all, and the water went on under them both.\n\n"
        '"You are late," said Marie.\n\n"The bells were early," said Tomas.\n\n"Then your watch is wrong."\n'
    )
    run = _run_text(tmp_path, "bridge.txt", text)
    genders = {entry.name: entry.gender for entry in run.bible.characters}
    assert genders["Tomas"] == "male" and genders["Marie"] != "male", "Tomas's pronoun does not gender Marie"


def test_pronoun_tag_prefers_the_subject_of_the_previous_sentence(tmp_path: Path) -> None:
    text = (
        "Halloran took the first watch while Brannock slept by the embers, and the wind moved in the dry grass beyond "
        "the camp for a long time before anything else did.\n\n"
        "Later, when the moon had gone behind the ridge, Halloran heard hooves on the road. He put his hand on "
        "Brannock\'s shoulder.\n\n"
        '"Riders," he whispered.\n\nBrannock was awake at once. "How many?"\n\n"Two. Maybe three."\n'
    )
    run = _run_text(tmp_path, "watch.txt", text)
    assert run.labels["c1p3s0"].speaker == "Halloran", "'he' is the subject Halloran, not the possessive object Brannock"
    assert run.labels["c1p4s1"].speaker == "Brannock" and run.labels["c1p5s0"].speaker == "Halloran"


def test_possessive_mention_still_resolves_a_pronoun_tag_without_a_subject(run: BookRun) -> None:
    assert run.labels["c2p10s0"].speaker == "Ansel Vey", "'...in Ansel\'s shaking hands.' then 'he said'"


def test_self_introduction_inside_a_quote_names_the_descriptor_entry(tmp_path: Path) -> None:
    text = (
        "The road ran north between two hedges, and Wren walked it alone with her pack on her back. At the crossroads "
        "a man was sitting on the milestone, whittling a stick, with a grey coat that had once been good.\n\n"
        '"You\'re a long way from anywhere," the man said.\n\n"So are you," said Wren.\n\n'
        '"True enough." The stranger put the stick away and stood. "Which way are you going?"\n\n"North. To Hallam."\n\n'
        '"Then we\'re going the same way. My name is Corwin. Corwin Tallow." He held out his hand.\n\n'
        'Wren took it. "Wren," she said.\n\n"You don\'t say much," Corwin said after a mile.\n\n"I don\'t have much to say."\n'
    )
    run = _run_text(tmp_path, "road.txt", text)
    names = {entry.name: entry for entry in run.bible.characters}
    assert set(names) == {"Wren", "Corwin Tallow"}, "one entry for the man, not 'The Man' plus 'Corwin'"
    assert {"the man", "the stranger", "Corwin"} <= set(names["Corwin Tallow"].aliases)
    assert names["Corwin Tallow"].gender == "male" and not names["Corwin Tallow"].provisional
    his_lines = [span_id for span_id in ("c1p2s0", "c1p4s0", "c1p4s2", "c1p6s0", "c1p8s0") if span_id in run.labels]
    assert {run.labels[span_id].speaker for span_id in his_lines} == {"Corwin Tallow"}


CHAPTER_FOUR = (
    "\n\nChapter 4: The Visitor\n\n"
    "Two days later a man in a black coat came up the cliff road on foot and stood at the cottage door without "
    "knocking. He was broad and grey-bearded and carried a leather case under one arm.\n\n"
    '"I\'m looking for the keeper," the man said.\n\n"You\'ve found her," said Mara.\n\n'
    '"Then I\'ll come in, if it\'s all the same to you. My name is Crane. I\'m from the Board."\n\n'
    'Mara did not move from the doorway.\n\n"The Board sent a letter," Mara said. "It said all it needed to say."\n\n'
    '"The letter said the light is closing," the man said. "It did not say what happens to the people in it."\n'
)


def test_descriptor_alias_of_an_earlier_character_does_not_reach_into_a_new_scene(sample_book_path: Path, tmp_path: Path) -> None:
    run = _run_text(tmp_path, "plus.txt", sample_book_path.read_text(encoding="utf-8").rstrip() + CHAPTER_FOUR)
    ansel = run.bible.find("Ansel Vey")
    assert ansel is not None and {"the man", "the stranger", "Ansel"} <= set(ansel.aliases), "the fixture's aliases are kept"
    crane_lines = [run.labels[i].speaker for i in ("c4p2s0", "c4p4s0", "c4p7s0", "c4p7s2")]
    assert "Ansel Vey" not in crane_lines, crane_lines
    assert len(set(crane_lines)) == 1
    crane = run.bible.find("Crane")
    assert crane is not None and crane.name == "Crane" and crane.line_count == 4
    assert ansel.line_count == 9, "Ansel keeps only his own lines"


def test_full_name_survives_a_chunk_cut_between_introduction_and_first_line(tmp_path: Path) -> None:
    intro = ARGUMENT.split("\n\n")[0]
    run = _run_text(tmp_path, "cut.txt", ARGUMENT, max_chars=len(intro) + 20)
    assert run.chunks[1][0].spans[-1].id.startswith("c1p1s"), "the introduction paragraph is a chunk of its own"
    dorian = run.bible.find("Dorian")
    assert dorian is not None and dorian.name == "Dorian Ash" and "Dorian" in dorian.aliases
    assert "Dorian Ash" in {entry.name for entry in run.bible.characters}
    stateless = HeuristicAnalyzer().analyze_chunk(run.chunks[1][1].model_copy(update={"context_before": ""}), CastBible())
    assert {u.name for u in stateless.characters if u.name.startswith("Dorian")} == {"Dorian"}, "without the context only the tag's single token is known"
