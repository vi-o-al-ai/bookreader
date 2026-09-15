"""bookreader.analysis.chunker - pack whole paragraphs of one chapter into analyzer-sized chunks.

Chunks never cross chapters and never split a paragraph: a paragraph that alone exceeds
``max_chars`` becomes its own chunk. Sizing counts paragraph text plus the two-character
separator that :attr:`bookreader.types.Chapter.text` puts between paragraphs. Each chunk
also lists which of its own paragraphs open a new scene (``Paragraph.scene_break_before``)
so analyzers can reset conversational state there.
"""
from __future__ import annotations

import logging

from bookreader.types import Chapter, Chunk, Paragraph

log = logging.getLogger(__name__)

CONTEXT_PARAGRAPHS = 2          # how many preceding paragraphs feed Chunk.context_before
PARAGRAPH_SEPARATOR_CHARS = 2   # "\n\n" between paragraphs, as in Chapter.text


def make_chunks(chapter: Chapter, max_chars: int) -> list[Chunk]:
    """Split *chapter* into chunks of whole paragraphs, each at most *max_chars* characters
    unless a single paragraph is longer. ``prior_mood`` is left at ``"none"``; the analyze
    stage sets it from the music action in force before each chunk. ``scene_break_paragraphs``
    holds the indices of the chunk's own paragraphs flagged ``scene_break_before``.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    groups: list[list[Paragraph]] = []
    current: list[Paragraph] = []
    size = 0
    for paragraph in chapter.paragraphs:
        length = len(paragraph.text)
        if current and size + PARAGRAPH_SEPARATOR_CHARS + length > max_chars:
            groups.append(current)
            current, size = [], 0
        if current:
            size += PARAGRAPH_SEPARATOR_CHARS
        current.append(paragraph)
        size += length
    if current:
        groups.append(current)

    chunks: list[Chunk] = []
    position = 0                                   # index into chapter.paragraphs of the group's first paragraph
    for chunk_index, group in enumerate(groups):
        context = chapter.paragraphs[max(0, position - CONTEXT_PARAGRAPHS):position]
        chunks.append(
            Chunk(
                chapter_index=chapter.index,
                chunk_index=chunk_index,
                paragraph_start=group[0].index,
                paragraph_end=group[-1].index,
                spans=[span for paragraph in group for span in paragraph.spans],
                context_before="\n\n".join(p.text for p in context),
                scene_break_paragraphs=[paragraph.index for paragraph in group if paragraph.scene_break_before],
            )
        )
        position += len(group)
    log.debug("chapter %d: %d paragraph(s) -> %d chunk(s) at <= %d chars", chapter.index, len(chapter.paragraphs), len(chunks), max_chars)
    return chunks
