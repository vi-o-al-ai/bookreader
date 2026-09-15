"""bookreader.ingest.spans - deterministic narration/quote splitting of paragraphs.

Quote style is detected once per book (:func:`detect_quote_style`) and every paragraph is
then split with :func:`split_spans`. Span texts are verbatim slices of the original paragraph
(curly quotes are only normalized in a working copy used to *find* the marks), so
``paragraph_text[span.start_char:span.end_char] == span.text`` always holds.
"""
from __future__ import annotations

import re
from typing import Literal

from bookreader.types import QuoteStyle, Span

MIN_STYLE_COUNT = 3

_DOUBLE_MARKS = '"“”„‟'
_SINGLE_MARKS = "'‘’‚‛"
_GUILLEMET_MARKS = "«»‹›"
_DASH_OPENER_RE = re.compile(r"^\s*(—|–|--)")
_DASH_SEPARATOR_RE = re.compile(r"(?<=\s)(—|–|--)(?=\s)")
_STYLE_ORDER: tuple[QuoteStyle, ...] = ("double", "single", "guillemet", "dash")

_SINGLE_OPEN_BEFORE = " \t\n([{—–-\"“"
_SINGLE_CLOSE_AFTER = " \t\n.,;:!?)]}\"”—–-"

# dash dialogue: '— You are late, said Marie.' carries its speech tag inside the dashed speech
_DASH_TAG_VERBS = (
    r"(?:said|says|asked|asks|replied|answered|shouted|whispered|muttered|murmured|cried|called|added|sighed|laughed|"
    r"snapped|growled|hissed|repeated|insisted|demanded|dijo|preguntó|respondió|gritó|murmuró|contestó|dit|demanda|"
    r"répondit|murmura|cria|ajouta|sagte|fragte|antwortete|rief|flüsterte|murmelte)"
)
_DASH_NAME = r"[A-ZÀ-ÖØ-Þ][\w'’-]*"
_DASH_TAG_RE = re.compile(
    rf",\s+(?:{_DASH_TAG_VERBS}\s+(?:{_DASH_NAME}|he|she|they|I|il|elle|er|sie|él|ella)\b|(?:{_DASH_NAME}|he|she|they|I|il|elle|er|sie|él|ella)\s+{_DASH_TAG_VERBS}\b)"
)
# 'Perhaps. He leaned on the parapet.' inside dashed speech that a later dash follows: the speech ends at the sentence
_DASH_SENTENCE_END_RE = re.compile(
    rf"[.!?][\"'”’)]*\s+(?=(?:he|she|they|He|She|They)\b|(?!(?:I|You|We|Yes|No|Not|Then|But|And|Or|So|If|What|Why|How|When|Where|Who|Oh|Ah)\b){_DASH_NAME}\s+[a-z])"
)

SpanKind = Literal["narration", "quote"]


def _paragraph_opener(paragraph: str) -> QuoteStyle | None:
    text = paragraph.lstrip()
    if not text:
        return None
    first = text[0]
    if first in _DOUBLE_MARKS:
        return "double"
    if first in _SINGLE_MARKS:
        return "single"
    if first in _GUILLEMET_MARKS:
        return "guillemet"
    if _DASH_OPENER_RE.match(text):
        return "dash"
    return None


def detect_quote_style(text: str) -> QuoteStyle:
    """Pick the book's dialogue convention from how paragraphs begin.

    Counts paragraphs opening with a double quote (straight or curly), a single quote, a
    guillemet or an em/en dash and returns the most frequent style, provided it occurs at
    least three times; ties and sparse books default to ``"double"``.
    """
    counts: dict[QuoteStyle, int] = {style: 0 for style in _STYLE_ORDER}
    for paragraph in re.split(r"\n\s*\n|\n", text):
        style = _paragraph_opener(paragraph)
        if style is not None:
            counts[style] += 1
    best: QuoteStyle = "double"
    for style in _STYLE_ORDER:
        if counts[style] > counts[best]:
            best = style
    return best if counts[best] >= MIN_STYLE_COUNT else "double"


def _normalize(text: str, marks: str, straight: str) -> str:
    return "".join(straight if c in marks else c for c in text)


def _double_boundaries(norm: str) -> list[int]:
    return [i for i, c in enumerate(norm) if c == '"']


def _guillemet_boundaries(norm: str) -> list[int]:
    return [i for i, c in enumerate(norm) if c == "«"]


def _single_boundaries(norm: str) -> list[int]:
    """Apostrophe-aware quote marks: openers only at start / after whitespace or open
    punctuation, closers only before whitespace / punctuation / end. An apostrophe that could be
    a closer but reads as a possessive or elision continuing the sentence (a letter before it, a
    space and a lower-case letter after it: "the Hardcastles' pride", "goin' home", "rock 'n'
    roll") keeps the quote open when a later closer exists before any later opener."""
    marks: list[int] = []
    open_quote = False
    positions = [i for i, c in enumerate(norm) if c == "'"]
    for n, i in enumerate(positions):
        before = norm[i - 1] if i > 0 else " "
        after = norm[i + 1] if i + 1 < len(norm) else " "
        if not open_quote and before in _SINGLE_OPEN_BEFORE:
            marks.append(i)
            open_quote = True
        elif open_quote and after in _SINGLE_CLOSE_AFTER:
            if before.isalpha() and after == " " and i + 2 < len(norm) and norm[i + 2].islower() and _later_closer(norm, positions[n + 1:]):
                continue
            marks.append(i)
            open_quote = False
    return marks


def _later_closer(norm: str, positions: list[int]) -> bool:
    """True when the next apostrophe that qualifies as a mark is a closer rather than an opener."""
    for j in positions:
        if norm[j - 1] in _SINGLE_OPEN_BEFORE:
            return False
        if (norm[j + 1] if j + 1 < len(norm) else " ") in _SINGLE_CLOSE_AFTER:
            return True
    return False


def _alternating_pieces(text: str, marks: list[int], mark_len: int = 1) -> list[tuple[SpanKind, int, int]]:
    """Cut *text* at *marks* into alternating narration/quote ranges (mark chars excluded).
    An unclosed quote runs to the end of the paragraph."""
    pieces: list[tuple[SpanKind, int, int]] = []
    pos = 0
    kind: SpanKind = "narration"
    for mark in marks:
        pieces.append((kind, pos, mark))
        pos = mark + mark_len
        kind = "quote" if kind == "narration" else "narration"
    pieces.append((kind, pos, len(text)))
    return pieces


def _dash_pieces(text: str) -> list[tuple[SpanKind, int, int]]:
    """A paragraph-initial dash opens speech that runs to the next ' — ' or the paragraph end.
    Within a speech piece a trailing speech tag (', said Marie.') is narration from its comma on,
    and when another dash follows, speech also ends at a sentence boundary that a narrative
    subject continues ('Perhaps. He leaned on the parapet. — Are you...'). Pieces alternate
    from the last one emitted, so the dash after a split-off narration piece opens speech again."""
    opener = _DASH_OPENER_RE.match(text)
    if opener is None:
        return [("narration", 0, len(text))]
    pieces: list[tuple[SpanKind, int, int]] = []
    pos = opener.end()
    kind: SpanKind = "quote"
    separators = list(_DASH_SEPARATOR_RE.finditer(text, pos))
    for index, end in enumerate([sep.start() for sep in separators] + [len(text)]):
        if kind == "quote":
            pieces.extend(_dash_speech(text, pos, end, followed_by_dash=index < len(separators)))
        else:
            pieces.append((kind, pos, end))
        if index < len(separators):
            pos = separators[index].end()
            kind = "narration" if pieces[-1][0] == "quote" else "quote"
    return pieces


def _dash_speech(text: str, start: int, end: int, *, followed_by_dash: bool) -> list[tuple[SpanKind, int, int]]:
    piece = text[start:end]
    cut = None
    if followed_by_dash:
        sentence = _DASH_SENTENCE_END_RE.search(piece)
        if sentence:
            cut = sentence.end()
    tag = _DASH_TAG_RE.search(piece)
    if tag and (cut is None or tag.start() < cut):
        cut = tag.start()
    if cut is None or not piece[:cut].strip():
        return [("quote", start, end)]
    return [("quote", start, start + cut), ("narration", start + cut, end)]


def _pieces(text: str, style: QuoteStyle) -> list[tuple[SpanKind, int, int]]:
    if style == "double":
        return _alternating_pieces(text, _double_boundaries(_normalize(text, _DOUBLE_MARKS, '"')))
    if style == "single":
        return _alternating_pieces(text, _single_boundaries(_normalize(text, _SINGLE_MARKS, "'")))
    if style == "guillemet":
        return _alternating_pieces(text, _guillemet_boundaries(_normalize(text, _GUILLEMET_MARKS, "«")))
    return _dash_pieces(text)


def split_spans(paragraph_text: str, chapter_index: int, paragraph_index: int, style: QuoteStyle) -> list[Span]:
    """Split one paragraph into alternating narration / quote spans.

    Span ids are ``c{chapter}p{paragraph}s{n}`` with ``n`` 0-based. Quote texts exclude the
    quote marks, every span text is stripped of surrounding whitespace with ``start_char`` /
    ``end_char`` tightened to match, and empty spans are dropped. Straight and curly marks of
    the chosen style are treated alike; an unclosed quote runs to the end of the paragraph. In
    single-quote style a possessive or elided apostrophe ("the boys' room", "goin' home") does
    not close a quote that a later apostrophe closes; in dash style a trailing speech tag and
    narration after a finished sentence are split out of the dashed speech (see ``_dash_pieces``).
    """
    spans: list[Span] = []
    for kind, start, end in _pieces(paragraph_text, style):
        piece = paragraph_text[start:end]
        stripped = piece.strip()
        if not stripped:
            continue
        lead = len(piece) - len(piece.lstrip())
        tight_start = start + lead
        spans.append(
            Span(
                id=f"c{chapter_index}p{paragraph_index}s{len(spans)}",
                kind=kind,
                text=stripped,
                start_char=tight_start,
                end_char=tight_start + len(stripped),
            )
        )
    return spans
