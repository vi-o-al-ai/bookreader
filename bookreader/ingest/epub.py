"""bookreader.ingest.epub - EPUB loader built on ebooklib and the stdlib HTMLParser."""
from __future__ import annotations

import logging
from html.parser import HTMLParser
from pathlib import Path
from typing import Literal

import ebooklib
from ebooklib import epub

from bookreader.ingest.text import collapse_whitespace
from bookreader.types import InputError

log = logging.getLogger(__name__)

MIN_FRONT_MATTER_WORDS = 100   # shorter items before the first chapter are nav/cover/toc
SCENE_BREAK_MARKER = "***"     # emitted for <hr>; turned into a flag by the text helpers

_BLOCK_TAGS = frozenset({"p", "div", "li", "blockquote", "h4", "h5", "h6", "section", "article", "aside",
                         "header", "footer", "nav", "figure", "figcaption", "table", "tr", "td", "th", "dd", "dt", "pre"})
_SPLIT_HEADING_TAGS = frozenset({"h1", "h2"})
_LINE_HEADING_TAGS = frozenset({"h3"})
_SKIP_TAGS = frozenset({"script", "style", "head", "title"})

BlockKind = Literal["heading", "text"]
Block = tuple[BlockKind, str]


class BlockExtractor(HTMLParser):
    """Collect block-level text from XHTML: ``("heading", title)`` for h1/h2 and
    ``("text", paragraph)`` for everything else (h3 becomes a plain heading line)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []
        self._buffer: list[str] = []
        self._kind: BlockKind = "text"
        self._skip_depth = 0

    def _flush(self) -> None:
        text = collapse_whitespace("".join(self._buffer))
        self._buffer = []
        if text:
            self.blocks.append((self._kind, text))
        self._kind = "text"

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag in _SPLIT_HEADING_TAGS:
            self._flush()
            self._kind = "heading"
        elif tag in _LINE_HEADING_TAGS or tag in _BLOCK_TAGS:
            self._flush()
        elif tag == "br":
            self._buffer.append(" ")
        elif tag == "hr":
            self._flush()
            self.blocks.append(("text", SCENE_BREAK_MARKER))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in _SKIP_TAGS:            # an empty <head/> or <script/> skips nothing
            self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag in _SPLIT_HEADING_TAGS or tag in _LINE_HEADING_TAGS or tag in _BLOCK_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._buffer.append(data)

    def close(self) -> None:
        super().close()
        self._flush()


def extract_blocks(html: str) -> list[Block]:
    """Parse one XHTML document into heading / text blocks in document order."""
    parser = BlockExtractor()
    parser.feed(html)
    parser.close()
    return parser.blocks


def _word_count(paragraphs: list[str]) -> int:
    return sum(len(p.split()) for p in paragraphs)


def _spine_documents(book: epub.EpubBook) -> list[epub.EpubHtml]:
    by_id = {item.get_id(): item for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)}
    docs = [by_id[idref] for idref, _linear in book.spine if idref in by_id]
    if not docs:
        docs = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
    return [doc for doc in docs if doc.is_chapter()]     # EPUB3 navigation documents are never content


def _decode(item: epub.EpubHtml) -> str:
    content = item.get_content()
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    return str(content)


def _book_title(book: epub.EpubBook) -> str | None:
    for value, _attrs in book.get_metadata("DC", "title"):
        if value and str(value).strip():
            return collapse_whitespace(str(value))
    return None


def epub_to_sections(path: Path) -> tuple[str | None, list[tuple[str, list[str]]]]:
    """Read an EPUB into ``(dc:title, [(chapter_title, paragraphs), ...])``.

    Documents are visited in spine order and split at their own h1/h2 headings; text after a
    heading-less document boundary continues the current chapter. Items under 100 words that
    precede the first chapter (nav, cover, toc) are dropped. When no document contains a
    heading, one untitled section (``""``) holding every paragraph is returned so the caller can
    fall back to :func:`bookreader.ingest.chapters.detect_chapters`.
    """
    try:
        book = epub.read_epub(str(path), options={"ignore_ncx": True})
    except Exception as exc:  # ebooklib raises zipfile/xml/KeyError variants for corrupt files
        raise InputError(f"cannot read EPUB {path.name}: {exc}") from exc

    documents = [extract_blocks(_decode(item)) for item in _spine_documents(book)]
    title = _book_title(book)
    has_headings = any(kind == "heading" for blocks in documents for kind, _ in blocks)

    if not has_headings:
        paragraphs: list[str] = []
        for blocks in documents:
            texts = [text for _, text in blocks]
            if not paragraphs and _word_count(texts) < MIN_FRONT_MATTER_WORDS:
                continue
            paragraphs.extend(texts)
        log.info("epub %s: no headings in %d document(s)", path.name, len(documents))
        return title, [("", paragraphs)]

    sections: list[tuple[str, list[str]]] = []
    front: list[str] = []
    for blocks in documents:
        item_words = _word_count([text for _, text in blocks])
        if not sections and item_words < MIN_FRONT_MATTER_WORDS:
            log.info("epub %s: dropping %d-word item before the first chapter", path.name, item_words)
            continue
        lead: list[str] = []
        for kind, text in blocks:
            if kind == "heading":
                sections.append((text, []))
            elif sections:
                sections[-1][1].append(text)
            else:
                lead.append(text)
        if lead:
            front.extend(lead)
    if front:
        sections.insert(0, ("Prologue", front))
    log.info("epub %s: %d document(s) -> %d section(s)", path.name, len(documents), len(sections))
    return title, sections
