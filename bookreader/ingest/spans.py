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
    punctuation, closers only before whitespace / punctuation / end."""
    marks: list[int] = []
    open_quote = False
    for i, c in enumerate(norm):
        if c != "'":
            continue
        before = norm[i - 1] if i > 0 else " "
        after = norm[i + 1] if i + 1 < len(norm) else " "
        if not open_quote and before in _SINGLE_OPEN_BEFORE:
            marks.append(i)
            open_quote = True
        elif open_quote and after in _SINGLE_CLOSE_AFTER:
            marks.append(i)
            open_quote = False
    return marks


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
    opener = _DASH_OPENER_RE.match(text)
    if opener is None:
        return [("narration", 0, len(text))]
    pieces: list[tuple[SpanKind, int, int]] = []
    pos = opener.end()
    kind: SpanKind = "quote"
    for sep in _DASH_SEPARATOR_RE.finditer(text, pos):
        pieces.append((kind, pos, sep.start()))
        pos = sep.end()
        kind = "narration" if kind == "quote" else "quote"
    pieces.append((kind, pos, len(text)))
    return pieces


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
    the chosen style are treated alike; an unclosed quote runs to the end of the paragraph.
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
