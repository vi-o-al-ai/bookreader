"""bookreader.ingest.chapters - heuristic chapter detection over a flat paragraph list."""
from __future__ import annotations

import logging
import math
import re

from bookreader.ingest.text import is_markdown_heading

log = logging.getLogger(__name__)

MAX_HEADING_CHARS = 80
MIN_HEADINGS = 2
MIN_CHAPTER_WORDS = 100        # shorter chapters are merged into the next one
MAX_PROLOGUE_FREE_WORDS = 100  # pre-heading leftovers above this become a 'Prologue'
SYNTHETIC_PART_WORDS = 6000    # target size of a synthetic 'Part N' when no headings are found
MAX_TITLE_WORDS = 12

_KEYWORD_RE = re.compile(r"^(chapter|part|book|prologue|epilogue|interlude)\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"^\d{1,4}\.?$")
_ROMAN_RE = re.compile(r"^(?=[IVXLCDM])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\.?$")
_MARKDOWN_MARKS_RE = re.compile(r"^#{1,6}\s+|\s+#+\s*$")


def word_count(text: str) -> int:
    """Number of whitespace-separated words in *text*."""
    return len(text.split())


def heading_title(paragraph: str) -> str | None:
    """Return the chapter title if *paragraph* looks like a heading, else ``None``.

    A heading is at most 80 characters and is one of: a line starting with
    chapter/part/book/prologue/epilogue/interlude, a markdown ``#`` heading, an ALL-CAPS
    line of 2..8 words, or a bare arabic/roman number. Lines containing quote marks are
    never headings (they are dialogue).
    """
    text = paragraph.strip()
    if not text or len(text) > MAX_HEADING_CHARS or '"' in text or "“" in text or "”" in text:
        return None
    if is_markdown_heading(text):
        return _MARKDOWN_MARKS_RE.sub("", text).strip()
    if _KEYWORD_RE.match(text) and word_count(text) <= MAX_TITLE_WORDS:
        return text
    if _NUMBER_RE.match(text) or _ROMAN_RE.match(text):
        return text
    words = word_count(text)
    if 2 <= words <= 8 and text.upper() == text and any(c.isalpha() for c in text):
        return text
    return None


def _is_title_candidate(paragraph: str) -> bool:
    text = paragraph.strip()
    return 0 < len(text) <= MAX_HEADING_CHARS and word_count(text) <= MAX_TITLE_WORDS


def _is_title_heading(title: str) -> bool:
    """A heading that names the book rather than a chapter: not 'Chapter 3', 'Part One' or '12'."""
    return not (_KEYWORD_RE.match(title) or _NUMBER_RE.match(title) or _ROMAN_RE.match(title))


def merge_short_chapters(chapters: list[tuple[str, list[str]]]) -> list[tuple[str, list[str]]]:
    """Merge every chapter shorter than 100 words into the chapter that follows it.

    The following chapter keeps its title and receives the short chapter's paragraphs at the
    front. Short chapters at the very end are appended to the last kept chapter; a book that
    is nothing but short chapters collapses into one chapter with the first title.
    """
    merged: list[tuple[str, list[str]]] = []
    pending: list[str] = []
    pending_title: str | None = None
    for title, paragraphs in chapters:
        if sum(word_count(p) for p in paragraphs) < MIN_CHAPTER_WORDS:
            if pending_title is None:
                pending_title = title
            pending.extend(paragraphs)
            continue
        merged.append((title, pending + paragraphs))
        pending = []
        pending_title = None
    if pending:
        if merged:
            last_title, last_paragraphs = merged[-1]
            merged[-1] = (last_title, last_paragraphs + pending)
        else:
            merged.append((pending_title or "Part 1", pending))
    return merged


def synthetic_parts(paragraphs: list[str]) -> list[tuple[str, list[str]]]:
    """Split *paragraphs* into evenly sized 'Part N' chapters of roughly 6000 words each."""
    total = sum(word_count(p) for p in paragraphs)
    n_parts = max(1, math.ceil(total / SYNTHETIC_PART_WORDS))
    target = total / n_parts
    parts: list[tuple[str, list[str]]] = []
    current: list[str] = []
    seen = 0
    for paragraph in paragraphs:
        current.append(paragraph)
        seen += word_count(paragraph)
        if len(parts) < n_parts - 1 and seen >= target * (len(parts) + 1):
            parts.append((f"Part {len(parts) + 1}", current))
            current = []
    if current or not parts:
        parts.append((f"Part {len(parts) + 1}", current))
    return parts


def detect_chapters(paragraphs: list[str]) -> tuple[str | None, list[tuple[str, list[str]]]]:
    """Split a flat paragraph list into ``(book_title, [(chapter_title, paragraphs), ...])``.

    With at least two headings (see :func:`heading_title`) each heading starts a chapter; the
    first short paragraph before the first heading is the book title (or the first heading
    itself when it has no body and does not look like a chapter label), further pre-heading text
    becomes a 'Prologue' when it exceeds 100 words (and is dropped otherwise), and chapters under
    100 words are merged into the next. With fewer than two headings nothing is dropped and the
    book is cut into synthetic ~6000-word 'Part N' chapters with no detected title.
    """
    heading_at = {i: t for i, p in enumerate(paragraphs) if (t := heading_title(p)) is not None}
    if len(heading_at) < MIN_HEADINGS:
        log.info("chapter detection: %d heading(s) found, using synthetic parts", len(heading_at))
        return None, synthetic_parts(paragraphs)

    first = min(heading_at)
    title: str | None = None
    leftover: list[str] = []
    for paragraph in paragraphs[:first]:
        if title is None and _is_title_candidate(paragraph):
            title = paragraph.strip()
        else:
            leftover.append(paragraph)

    chapters: list[tuple[str, list[str]]] = []
    if sum(word_count(p) for p in leftover) > MAX_PROLOGUE_FREE_WORDS:
        chapters.append(("Prologue", leftover))
    elif leftover:
        log.info("chapter detection: dropping %d short pre-heading paragraph(s)", len(leftover))

    current_title = heading_at[first]
    body: list[str] = []
    for i in range(first + 1, len(paragraphs)):
        if i in heading_at:
            chapters.append((current_title, body))
            current_title, body = heading_at[i], []
        else:
            body.append(paragraphs[i])
    chapters.append((current_title, body))
    if title is None and not chapters[0][1] and len(chapters) > 1 and _is_title_heading(chapters[0][0]):
        # '# The Book' followed directly by '## Chapter 1': the empty first heading is the title
        title = chapters.pop(0)[0]
    merged = merge_short_chapters(chapters)
    log.info("chapter detection: %d heading(s) -> %d chapter(s)", len(heading_at), len(merged))
    return title, merged
