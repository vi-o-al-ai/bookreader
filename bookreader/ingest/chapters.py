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

_NUMBER_WORDS = (
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
    r"seventeen|eighteen|nineteen|(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:-(?:one|two|three|"
    r"four|five|six|seven|eight|nine))?|hundred|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last)"
)
# 'Chapter 1: The Storm Bell', 'PART TWO', 'Book II - The Return', 'Chapter 3.'; the keyword must be followed by
# a designator (number, roman numeral or number word) so 'Part of her wanted to run.' is prose, not a heading
_KEYWORD_RE = re.compile(
    rf"^(?:chapter|part|book)\s+(?P<designator>\d{{1,4}}|[ivxlcdm]{{1,9}}|{_NUMBER_WORDS})\b\.?(?:\s*[:.\-–—]\s*\S.*|\s+\S.*)?$"
    r"|^(?:prologue|epilogue|interlude)\b\.?(?:\s*[:.\-–—]\s*\S.*)?$",
    re.IGNORECASE,
)
_NUMBER_WORD_RE = re.compile(_NUMBER_WORDS, re.IGNORECASE)
_NUMBER_RE = re.compile(r"^\d{1,4}\.?$")
_ROMAN_RE = re.compile(r"^(?=[IVXLCDM])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\.?$")
_MARKDOWN_MARKS_RE = re.compile(r"^#{1,6}\s+|\s+#+\s*$")
_SENTENCE_PUNCTUATION = ".!?,;:"


def word_count(text: str) -> int:
    """Number of whitespace-separated words in *text*."""
    return len(text.split())


def _keyword_heading(text: str) -> bool:
    """'Chapter 1', 'Part One', 'BOOK II: The Return', 'Prologue', 'Epilogue - Ten Years Later'; a
    chapter/part/book keyword needs a designator (digits, an upper-case roman numeral or a number
    word), and prologue/epilogue/interlude may only be followed by a separated subtitle, so a
    sentence that merely starts with one of the words ('Book me a room at the inn.') is prose."""
    match = _KEYWORD_RE.match(text)
    if match is None:
        return False
    designator = match.group("designator")
    if designator and designator.isalpha() and not _NUMBER_WORD_RE.fullmatch(designator) and not _ROMAN_RE.match(designator):
        return False       # 'Part civil' is not 'Part CIVIL'
    return True


def _caps_heading(text: str) -> bool:
    """An ALL-CAPS line of 2..8 words with no sentence punctuation at the end and no quote marks:
    'THE STORM BELL' yes, 'I SAID NO.' and "'KEEP OUT'" no."""
    words = word_count(text)
    if not (2 <= words <= 8 and text.upper() == text and any(c.isalpha() for c in text)):
        return False
    return text[-1] not in _SENTENCE_PUNCTUATION and not any(c in text for c in "'‘’")


def heading_title(paragraph: str) -> str | None:
    """Return the chapter title if *paragraph* looks like a heading, else ``None``.

    A heading is at most 80 characters and is one of: chapter/part/book followed by a number,
    roman numeral or number word (plus an optional subtitle), prologue/epilogue/interlude alone or
    with a separated subtitle, a markdown ``#`` heading, an ALL-CAPS line of 2..8 words that does
    not end in sentence punctuation, or a bare arabic/roman number. Lines containing quote marks
    are never headings (they are dialogue).
    """
    text = paragraph.strip()
    if not text or len(text) > MAX_HEADING_CHARS or '"' in text or "“" in text or "”" in text:
        return None
    if is_markdown_heading(text):
        return _MARKDOWN_MARKS_RE.sub("", text).strip()
    if _keyword_heading(text) and word_count(text) <= MAX_TITLE_WORDS:
        return text
    if _NUMBER_RE.match(text) or _ROMAN_RE.match(text):
        return text
    if _caps_heading(text):
        return text
    return None


def is_strong_heading(paragraph: str) -> bool:
    """A heading that cannot be mistaken for narrative text: keyworded, numbered or markdown."""
    text = paragraph.strip()
    return heading_title(text) is not None and not _caps_heading(text)


def _is_title_candidate(paragraph: str) -> bool:
    text = paragraph.strip()
    return 0 < len(text) <= MAX_HEADING_CHARS and word_count(text) <= MAX_TITLE_WORDS


def _is_title_heading(title: str) -> bool:
    """A heading that names the book rather than a chapter: not 'Chapter 3', 'Part One' or '12'."""
    return not (_keyword_heading(title) or _NUMBER_RE.match(title) or _ROMAN_RE.match(title))


def _heading_map(paragraphs: list[str]) -> dict[int, str]:
    """Index -> title of every heading paragraph. When the book has at least two strong headings
    (keyworded, numbered or markdown) a bare ALL-CAPS line counts as a heading only directly after
    another heading (a subtitle) or as the book's last line ('THE END'); elsewhere it is narrative
    set in capitals (a sign, a telegram, a shouted line) and stays a spoken paragraph."""
    heading_at = {i: t for i, p in enumerate(paragraphs) if (t := heading_title(p)) is not None}
    strong = {i for i in heading_at if is_strong_heading(paragraphs[i])}
    if len(strong) < MIN_HEADINGS:
        return heading_at
    kept: dict[int, str] = {}
    for i in sorted(heading_at):
        if i in strong or i - 1 in kept or i == len(paragraphs) - 1:      # a subtitle, or a trailer such as 'THE END'
            kept[i] = heading_at[i]
        else:
            log.info("chapter detection: keeping ALL-CAPS line %r as text (the book has %d keyworded headings)", heading_at[i], len(strong))
    return kept


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
    becomes a 'Prologue' when it exceeds 100 words or contains a paragraph longer than a title
    line (title-length front matter such as 'by Someone' is dropped with a warning), and chapters
    under 100 words are merged into the next. With fewer than two headings nothing is dropped and the
    book is cut into synthetic ~6000-word 'Part N' chapters with no detected title.
    """
    heading_at = _heading_map(paragraphs)
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
    prose = any(word_count(p) > MAX_TITLE_WORDS for p in leftover)
    if leftover and (prose or sum(word_count(p) for p in leftover) > MAX_PROLOGUE_FREE_WORDS):
        chapters.append(("Prologue", leftover))       # merged into the first chapter when short; never dropped
    elif leftover:
        log.warning("chapter detection: dropping %d short pre-heading line(s) as front matter: %s", len(leftover), "; ".join(repr(p.strip()[:40]) for p in leftover))

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
