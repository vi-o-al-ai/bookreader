"""Chunker: whole-paragraph packing, oversized paragraphs, chapter isolation, context_before."""
from __future__ import annotations

from pathlib import Path

import pytest

from bookreader.analysis.chunker import make_chunks
from bookreader.ingest import load_book
from bookreader.types import Book, Chapter, Paragraph, Span


def _chapter(lengths: list[int], index: int = 1) -> Chapter:
    """A chapter whose paragraph i is a narration span of exactly lengths[i] characters."""
    paragraphs = []
    for i, length in enumerate(lengths, start=1):
        text = ("word " * (length // 5 + 1))[:length]
        paragraphs.append(
            Paragraph(index=i, text=text, spans=[Span(id=f"c{index}p{i}s0", kind="narration", text=text, start_char=0, end_char=length)])
        )
    return Chapter(index=index, title=f"Chapter {index}", paragraphs=paragraphs)


@pytest.fixture(scope="module")
def book(sample_book_path: Path) -> Book:
    return load_book(sample_book_path)


# --------------------------------------------------------------------------- paragraph bounds
@pytest.mark.parametrize("max_chars", [120, 300, 800, 6000])
def test_chunks_cover_paragraphs_contiguously(book: Book, max_chars: int) -> None:
    for chapter in book.chapters:
        chunks = make_chunks(chapter, max_chars)
        assert chunks, "every chapter yields at least one chunk"
        assert chunks[0].paragraph_start == 1
        assert chunks[-1].paragraph_end == len(chapter.paragraphs)
        for previous, current in zip(chunks, chunks[1:]):
            assert current.paragraph_start == previous.paragraph_end + 1
        for k, chunk in enumerate(chunks):
            assert chunk.chunk_index == k
            assert chunk.prior_mood == "none"
            paragraphs = chapter.paragraphs[chunk.paragraph_start - 1:chunk.paragraph_end]
            assert chunk.spans == [span for p in paragraphs for span in p.spans]
            size = sum(len(p.text) for p in paragraphs) + 2 * (len(paragraphs) - 1)
            assert size <= max_chars or len(paragraphs) == 1, "only a lone oversized paragraph may exceed max_chars"


def test_large_limit_gives_one_chunk_per_chapter(book: Book) -> None:
    for chapter in book.chapters:
        chunks = make_chunks(chapter, 6000)
        assert len(chunks) == 1
        assert chunks[0].spans == chapter.spans
        assert chunks[0].text == " ".join(s.text for s in chapter.spans)


def test_packing_is_greedy() -> None:
    chapter = _chapter([100, 100, 100, 100])
    chunks = make_chunks(chapter, 202)          # 100 + 2 + 100 fits, a third paragraph does not
    assert [(c.paragraph_start, c.paragraph_end) for c in chunks] == [(1, 2), (3, 4)]


# --------------------------------------------------------------------------- oversized paragraph
def test_oversized_paragraph_becomes_its_own_chunk() -> None:
    chapter = _chapter([100, 5000, 100, 100])
    chunks = make_chunks(chapter, 300)
    assert [(c.paragraph_start, c.paragraph_end) for c in chunks] == [(1, 1), (2, 2), (3, 4)]
    big = chunks[1]
    assert len(big.spans) == 1 and len(big.spans[0].text) == 5000, "spans are never re-split"


def test_oversized_first_paragraph() -> None:
    chapter = _chapter([900, 50])
    chunks = make_chunks(chapter, 100)
    assert [(c.paragraph_start, c.paragraph_end) for c in chunks] == [(1, 1), (2, 2)]


def test_invalid_limit() -> None:
    with pytest.raises(ValueError):
        make_chunks(_chapter([10]), 0)


def test_empty_chapter() -> None:
    assert make_chunks(Chapter(index=1, title="Empty", paragraphs=[]), 100) == []


# --------------------------------------------------------------------------- chapter isolation
def test_chunks_never_cross_chapters(book: Book) -> None:
    all_spans = []
    for chapter in book.chapters:
        for chunk in make_chunks(chapter, 200):
            assert chunk.chapter_index == chapter.index
            assert all(span.id.startswith(f"c{chapter.index}p") for span in chunk.spans)
            all_spans.extend(chunk.spans)
    assert all_spans == [span for chapter in book.chapters for span in chapter.spans]


def test_chunking_one_chapter_ignores_the_others(book: Book) -> None:
    second = book.chapters[1]
    chunks = make_chunks(second, 250)
    assert len(chunks) > 1
    assert chunks[0].context_before == "", "context never reaches back into the previous chapter"
    assert {c.chapter_index for c in chunks} == {2}


# --------------------------------------------------------------------------- context_before
def test_context_before_is_previous_two_paragraphs(book: Book) -> None:
    chapter = book.chapters[1]
    chunks = make_chunks(chapter, 300)
    assert len(chunks) >= 3
    assert chunks[0].context_before == ""
    for chunk in chunks[1:]:
        first = chunk.paragraph_start
        expected = [p.text for p in chapter.paragraphs[max(0, first - 3):first - 1]]
        assert chunk.context_before == "\n\n".join(expected)
        assert len(expected) == min(2, first - 1)


def test_context_before_single_paragraph_when_chunk_starts_at_second_paragraph() -> None:
    chapter = _chapter([100, 100, 100])
    chunks = make_chunks(chapter, 100)
    assert [(c.paragraph_start, c.paragraph_end) for c in chunks] == [(1, 1), (2, 2), (3, 3)]
    assert chunks[1].context_before == chapter.paragraphs[0].text
    assert chunks[2].context_before == chapter.paragraphs[0].text + "\n\n" + chapter.paragraphs[1].text


def test_context_is_not_part_of_chunk_text() -> None:
    chapter = _chapter([50, 50])
    chunks = make_chunks(chapter, 50)
    assert chunks[1].text == chapter.paragraphs[1].text
    assert chunks[1].context_before == chapter.paragraphs[0].text
