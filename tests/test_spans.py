"""Span splitting: quote-style detection and narration/quote alternation."""
from __future__ import annotations

from pathlib import Path

from bookreader.ingest import load_book
from bookreader.ingest.spans import detect_quote_style, split_spans
from bookreader.types import Span


def pairs(spans: list[Span]) -> list[tuple[str, str]]:
    return [(s.kind, s.text) for s in spans]


def assert_offsets(text: str, spans: list[Span]) -> None:
    for s in spans:
        assert text[s.start_char:s.end_char] == s.text
        assert s.text == s.text.strip() and s.text
    for a, b in zip(spans, spans[1:]):
        assert a.end_char <= b.start_char


# --------------------------------------------------------------------------- fixture
def test_fixture_paragraph_splits_into_quote_narration_quote(sample_book_path: Path) -> None:
    book = load_book(sample_book_path)
    paragraph = book.chapters[0].paragraphs[1]
    assert paragraph.text == '"Tobias!" she shouted up the stairwell. "The bell! Ring the bell!"'
    assert pairs(paragraph.spans) == [
        ("quote", "Tobias!"),
        ("narration", "she shouted up the stairwell."),
        ("quote", "The bell! Ring the bell!"),
    ]
    assert [s.id for s in paragraph.spans] == ["c1p2s0", "c1p2s1", "c1p2s2"]
    assert_offsets(paragraph.text, paragraph.spans)


def test_fixture_every_span_indexes_its_paragraph(sample_book_path: Path) -> None:
    book = load_book(sample_book_path)
    for chapter in book.chapters:
        for paragraph in chapter.paragraphs:
            assert_offsets(paragraph.text, paragraph.spans)
            for n, span in enumerate(paragraph.spans):
                assert span.id == f"c{chapter.index}p{paragraph.index}s{n}"
    only_quote = book.chapters[1].paragraphs[10]
    assert only_quote.text == '"We did."'
    assert pairs(only_quote.spans) == [("quote", "We did.")]
    narration_only = book.chapters[0].paragraphs[0]
    assert len(narration_only.spans) == 1 and narration_only.spans[0].kind == "narration"


# --------------------------------------------------------------------------- double quotes
def test_ids_are_stable_and_zero_based() -> None:
    text = 'He said, "Go." She went.'
    first = split_spans(text, 3, 12, "double")
    assert [s.id for s in first] == ["c3p12s0", "c3p12s1", "c3p12s2"]
    assert first == split_spans(text, 3, 12, "double")
    assert pairs(first) == [("narration", "He said,"), ("quote", "Go."), ("narration", "She went.")]
    assert_offsets(text, first)


def test_unbalanced_quote_runs_to_paragraph_end() -> None:
    text = 'Mara turned. "I never said that, and I never will'
    spans = split_spans(text, 1, 1, "double")
    assert pairs(spans) == [("narration", "Mara turned."), ("quote", "I never said that, and I never will")]
    assert spans[-1].end_char == len(text)
    assert_offsets(text, spans)


def test_curly_double_quotes_and_inner_apostrophes() -> None:
    text = "“I can’t reach the rope, it’s jammed!” Tobias cried. “Help!”"
    spans = split_spans(text, 2, 5, "double")
    assert pairs(spans) == [
        ("quote", "I can’t reach the rope, it’s jammed!"),
        ("narration", "Tobias cried."),
        ("quote", "Help!"),
    ]
    assert_offsets(text, spans)
    assert "“" not in "".join(s.text for s in spans)


def test_empty_and_whitespace_spans_are_dropped() -> None:
    text = '"" "Hello."   "" '
    spans = split_spans(text, 1, 1, "double")
    assert pairs(spans) == [("quote", "Hello.")]
    assert split_spans("   ", 1, 1, "double") == []
    assert split_spans("", 1, 1, "double") == []


def test_narration_offsets_are_tightened_to_stripped_text() -> None:
    text = '  Leading space.  "Quote."  trailing  '
    spans = split_spans(text, 1, 1, "double")
    assert pairs(spans) == [("narration", "Leading space."), ("quote", "Quote."), ("narration", "trailing")]
    assert spans[0].start_char == 2
    assert_offsets(text, spans)


# --------------------------------------------------------------------------- style detection
def test_detect_quote_style_counts_paragraph_openers() -> None:
    double = '"One."\n\n"Two."\n\n"Three."\n\nNarration.'
    assert detect_quote_style(double) == "double"
    curly_double = "“One.”\n\n“Two.”\n\n“Three.”"
    assert detect_quote_style(curly_double) == "double"
    single = "'One.'\n\n'Two.'\n\n‘Three.’\n\n\"Just one double.\""
    assert detect_quote_style(single) == "single"
    guillemet = "«Un.»\n\n«Deux.»\n\n«Trois.»\n\n«Quatre.»"
    assert detect_quote_style(guillemet) == "guillemet"
    dash = "— Uno.\n\n– Dos.\n\n-- Tres.\n\nNarration."
    assert detect_quote_style(dash) == "dash"


def test_detect_quote_style_defaults_to_double() -> None:
    assert detect_quote_style("No dialogue here at all.\n\nStill none.") == "double"
    assert detect_quote_style("'One.'\n\n'Two.'") == "double"          # below the minimum of 3
    assert detect_quote_style("") == "double"


# --------------------------------------------------------------------------- single quotes
def test_single_quote_book_with_apostrophes(tmp_path: Path) -> None:
    body = " ".join(["The keeper watched the sea and waited for the morning light to come."] * 10)
    paragraphs = [
        "'I can't reach the rope,' said Tobias. 'It's jammed!'",
        "'Tobias's lantern is out,' Mara said, 'and the boy's boots are wet.'",
        "'Stand back.' She drew her knife.",
        "Mara didn't answer; the sea's roar was too loud. 'They heard,' she said at last.",
    ]
    path = tmp_path / "single.txt"
    path.write_text("Title\n\nChapter 1\n\n" + "\n\n".join(paragraphs) + "\n\n" + body + "\n\nChapter 2\n\n" + body, encoding="utf-8")
    book = load_book(path)
    assert book.quote_style == "single"
    ch = book.chapters[0]
    assert pairs(ch.paragraphs[0].spans) == [("quote", "I can't reach the rope,"), ("narration", "said Tobias."), ("quote", "It's jammed!")]
    assert pairs(ch.paragraphs[1].spans) == [
        ("quote", "Tobias's lantern is out,"), ("narration", "Mara said,"), ("quote", "and the boy's boots are wet."),
    ]
    assert pairs(ch.paragraphs[2].spans) == [("quote", "Stand back."), ("narration", "She drew her knife.")]
    assert pairs(ch.paragraphs[3].spans) == [
        ("narration", "Mara didn't answer; the sea's roar was too loud."), ("quote", "They heard,"), ("narration", "she said at last."),
    ]
    for p in ch.paragraphs:
        assert_offsets(p.text, p.spans)


def test_single_quote_possessives_and_elisions_do_not_close_the_quote() -> None:
    cases = {
        "'Well, come and look at it, then. It's the Hardcastles' pride and joy, or it was.'": [
            ("quote", "Well, come and look at it, then. It's the Hardcastles' pride and joy, or it was."),
        ],
        "'I've seen the boys' room,' said Nell. 'It's a mess.'": [
            ("quote", "I've seen the boys' room,"), ("narration", "said Nell."), ("quote", "It's a mess."),
        ],
        "'Rock 'n' roll,' he said.": [("quote", "Rock 'n' roll,"), ("narration", "he said.")],
        "'I'm goin' home,' she said.": [("quote", "I'm goin' home,"), ("narration", "she said.")],
        "'My parents' house is over there,' said Tom.": [("quote", "My parents' house is over there,"), ("narration", "said Tom.")],
    }
    for text, expected in cases.items():
        spans = split_spans(text, 1, 1, "single")
        assert pairs(spans) == expected, text
        assert_offsets(text, spans)
    unbalanced = "'The boys' room is a mess"
    assert pairs(split_spans(unbalanced, 1, 1, "single")) == [("quote", "The boys"), ("narration", "room is a mess")], \
        "with no later closer the old rule still applies"


def test_single_quote_curly_and_unbalanced() -> None:
    text = "‘Where’s the boy?’ she asked. ‘I don’t know"
    spans = split_spans(text, 1, 1, "single")
    assert pairs(spans) == [("quote", "Where’s the boy?"), ("narration", "she asked."), ("quote", "I don’t know")]
    assert_offsets(text, spans)


# --------------------------------------------------------------------------- guillemets
def test_guillemets() -> None:
    text = "«Viens ici», dit-elle. «Maintenant !»"
    spans = split_spans(text, 1, 1, "guillemet")
    assert pairs(spans) == [("quote", "Viens ici"), ("narration", ", dit-elle."), ("quote", "Maintenant !")]
    assert_offsets(text, spans)
    reversed_style = "»Komm her«, sagte sie."
    assert pairs(split_spans(reversed_style, 1, 1, "guillemet")) == [("quote", "Komm her"), ("narration", ", sagte sie.")]
    assert pairs(split_spans("«Unclosed", 1, 1, "guillemet")) == [("quote", "Unclosed")]


# --------------------------------------------------------------------------- dash dialogue
def test_dash_dialogue() -> None:
    text = "— Ven aquí — dijo ella — ahora mismo."
    spans = split_spans(text, 1, 1, "dash")
    assert pairs(spans) == [("quote", "Ven aquí"), ("narration", "dijo ella"), ("quote", "ahora mismo.")]
    assert_offsets(text, spans)
    to_end = "-- Nobody is coming."
    assert pairs(split_spans(to_end, 1, 1, "dash")) == [("quote", "Nobody is coming.")]
    narration = "The sea — grey and flat — lay silent."
    assert pairs(split_spans(narration, 1, 1, "dash")) == [("narration", narration)]
    assert_offsets(narration, split_spans(narration, 1, 1, "dash"))


def test_dash_dialogue_splits_out_speech_tags_and_narration() -> None:
    cases = {
        "— You are late, said Marie.": [("quote", "You are late"), ("narration", ", said Marie.")],
        "— I always pay, said Tomas.": [("quote", "I always pay"), ("narration", ", said Tomas.")],
        "— The bells were early, said Tomas. — I set my watch by them.": [
            ("quote", "The bells were early"), ("narration", ", said Tomas."), ("quote", "I set my watch by them."),
        ],
        "— Perhaps. He leaned on the parapet beside her. — Are you going to be angry all evening?": [
            ("quote", "Perhaps."), ("narration", "He leaned on the parapet beside her."), ("quote", "Are you going to be angry all evening?"),
        ],
        "— Decide quickly. The café closes at nine.": [("quote", "Decide quickly. The café closes at nine.")],
        "— Fine. But you are paying.": [("quote", "Fine. But you are paying.")],
        "— Two of the usual, said Tomas, and sat down. — And bread.": [
            ("quote", "Two of the usual"), ("narration", ", said Tomas, and sat down."), ("quote", "And bread."),
        ],
    }
    for text, expected in cases.items():
        spans = split_spans(text, 1, 1, "dash")
        assert pairs(spans) == expected, text
        assert_offsets(text, spans)


def test_other_styles_marks_are_left_alone() -> None:
    text = "«Not a quote here», he said. 'Nor this.'"
    assert pairs(split_spans(text, 1, 1, "double")) == [("narration", text)]
    text = "— Not dash dialogue — in double mode."
    assert pairs(split_spans(text, 1, 1, "double")) == [("narration", text)]
