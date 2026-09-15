"""bookreader.ingest - turn an uploaded file into a :class:`bookreader.types.Book`.

``load_book`` dispatches on the file suffix (.txt/.md, .epub, .pdf), detects chapters and the
book's quote style, and splits every paragraph into narration/quote spans. Chapter titles are
never emitted as paragraphs or spans.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from bookreader.ingest.chapters import detect_chapters, merge_short_chapters, word_count
from bookreader.ingest.epub import epub_to_sections
from bookreader.ingest.pdf import pdf_to_text
from bookreader.ingest.spans import detect_quote_style, split_spans
from bookreader.ingest.text import apply_scene_markers, text_to_paragraphs
from bookreader.types import Book, Chapter, InputError, Paragraph

__all__ = ["load_book", "MIN_BOOK_WORDS", "SUPPORTED_SUFFIXES"]

log = logging.getLogger(__name__)

MIN_BOOK_WORDS = 50
SUPPORTED_SUFFIXES: tuple[str, ...] = (".txt", ".md", ".epub", ".pdf")
_TEXT_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "cp1252")

Sections = list[tuple[str, list[str]]]


def _decode_text(data: bytes) -> str:
    for encoding in _TEXT_ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")


def _text_sections(raw: str) -> tuple[str | None, Sections, dict[int, bool]]:
    """Text path: paragraphs -> chapters, keeping scene-break flags keyed by paragraph position."""
    blocks = text_to_paragraphs(raw)
    title, sections = detect_chapters([text for text, _ in blocks])
    flags = _align_flags(blocks, sections)
    return title, sections, flags


def _align_flags(blocks: list[tuple[str, bool]], sections: Sections) -> dict[int, bool]:
    """Chapter detection preserves paragraph order and only drops paragraphs, so the chapter
    paragraphs are a subsequence of *blocks*; walk both to recover scene-break flags."""
    flags: dict[int, bool] = {}
    cursor = 0
    position = 0
    for _, paragraphs in sections:
        for text in paragraphs:
            while cursor < len(blocks) and blocks[cursor][0] != text:
                cursor += 1
            flags[position] = blocks[cursor][1] if cursor < len(blocks) else False
            cursor += 1
            position += 1
    return flags


def _epub_sections(path: Path) -> tuple[str | None, Sections, dict[int, bool]]:
    """EPUB path: sections from the loader, or chapter detection when it found no headings."""
    title, sections = epub_to_sections(path)
    if len(sections) == 1 and sections[0][0] == "":
        blocks = apply_scene_markers([(p, False) for p in sections[0][1]])
        detected, chapters = detect_chapters([text for text, _ in blocks])
        return title or detected, chapters, _align_flags(blocks, chapters)
    blocks: list[tuple[str, bool]] = []
    cleaned: Sections = []
    for chapter_title, paragraphs in sections:
        chapter_blocks = apply_scene_markers([(p, False) for p in paragraphs])
        blocks.extend(chapter_blocks)
        cleaned.append((chapter_title, [text for text, _ in chapter_blocks]))
    merged = merge_short_chapters(cleaned)
    return title, merged, _align_flags(blocks, merged)


def load_book(path: Path, title_hint: str | None = None) -> Book:
    """Read *path* and return a fully split :class:`Book`.

    Raises :class:`InputError` for unsupported suffixes, unreadable files and books shorter
    than 50 words. ``Book.title`` is the detected title, else *title_hint*, else the file stem.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise InputError(f"unsupported file type '{suffix or path.name}'; supported: {', '.join(SUPPORTED_SUFFIXES)}")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise InputError(f"cannot read {path.name}: {exc}") from exc
    sha256 = hashlib.sha256(data).hexdigest()

    if suffix == ".epub":
        detected, sections, flags = _epub_sections(path)
    elif suffix == ".pdf":
        detected, sections, flags = _text_sections(pdf_to_text(path))
    else:
        detected, sections, flags = _text_sections(_decode_text(data))

    total_words = sum(word_count(p) for _, paragraphs in sections for p in paragraphs)
    if total_words < MIN_BOOK_WORDS:
        raise InputError(f"{path.name} contains only {total_words} words; a book needs at least {MIN_BOOK_WORDS}")

    quote_style = detect_quote_style("\n\n".join(p for _, paragraphs in sections for p in paragraphs))
    chapters: list[Chapter] = []
    position = 0
    for chapter_index, (chapter_title, paragraphs) in enumerate(sections, start=1):
        items: list[Paragraph] = []
        for paragraph_index, text in enumerate(paragraphs, start=1):
            items.append(
                Paragraph(
                    index=paragraph_index,
                    text=text,
                    scene_break_before=flags.get(position, False),
                    spans=split_spans(text, chapter_index, paragraph_index, quote_style),
                )
            )
            position += 1
        chapters.append(Chapter(index=chapter_index, title=chapter_title, paragraphs=items))

    title = detected or title_hint or path.stem
    log.info("loaded %s: %r, %d chapter(s), %d words, quote style %s", path.name, title, len(chapters), total_words, quote_style)
    return Book(
        title=title,
        source_filename=path.name,
        source_sha256=sha256,
        quote_style=quote_style,
        chapters=chapters,
        word_count=total_words,
    )
