"""bookreader.ingest.text - plain text / markdown normalization into paragraphs.

The text path is also the final step of the PDF path: once a PDF has been turned into
plain text it flows through :func:`text_to_paragraphs` like an uploaded ``.txt``.
"""
from __future__ import annotations

import re

_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s+\S")
_SCENE_MARKER_RE = re.compile(r"^(?:(?:[*•·]\s*){3,}|[-_—–]{3,}|(?:#\s*){3,})$")


def normalize_text(raw: str) -> str:
    """Strip a BOM and normalize CRLF / CR line endings to LF."""
    text = raw.lstrip("﻿")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def is_markdown_heading(line: str) -> bool:
    """True for an ATX markdown heading line such as ``## Chapter 2``."""
    return bool(_MARKDOWN_HEADING_RE.match(line.strip()))


def is_scene_marker(paragraph: str) -> bool:
    """True when a whole paragraph is a scene-break marker (``***``, ``* * *``, ``---`` ...)."""
    return bool(_SCENE_MARKER_RE.match(paragraph.strip()))


def collapse_whitespace(text: str) -> str:
    """Join hard-wrapped lines and collapse runs of whitespace into single spaces."""
    return " ".join(text.split())


def split_blocks(text: str) -> list[tuple[str, bool]]:
    """Split normalized text into ``(paragraph, scene_break_before)`` blocks.

    Paragraphs are separated by blank lines; two or more blank lines in a row mark a scene
    break before the following paragraph. Markdown heading lines always form a block of
    their own so they can be recognized by chapter detection even without blank lines.
    """
    blocks: list[tuple[str, bool]] = []
    current: list[str] = []
    blank_run = 0
    pending_break = False

    def flush() -> None:
        nonlocal current, pending_break
        if current:
            blocks.append((collapse_whitespace(" ".join(current)), pending_break))
            pending_break = False
        current = []

    for line in text.split("\n"):
        if not line.strip():
            blank_run += 1
            flush()
            continue
        if blank_run >= 2:
            pending_break = True
        blank_run = 0
        if is_markdown_heading(line):
            flush()
            current = [line.strip()]
            flush()
            continue
        current.append(line.strip())
    flush()
    return blocks


def apply_scene_markers(blocks: list[tuple[str, bool]]) -> list[tuple[str, bool]]:
    """Remove marker paragraphs (``***`` ...) and flag the paragraph that follows them."""
    out: list[tuple[str, bool]] = []
    pending = False
    for text, flag in blocks:
        if is_scene_marker(text):
            pending = True
            continue
        out.append((text, flag or pending))
        pending = False
    return out


def text_to_paragraphs(raw: str) -> list[tuple[str, bool]]:
    """Turn raw ``.txt`` / ``.md`` content into ``(paragraph_text, scene_break_before)`` pairs.

    Newlines are normalized, hard-wrapped lines joined, markdown headings kept as their own
    paragraphs (with the ``#`` marks, so chapter detection can see them) and scene-break
    marker paragraphs converted into a flag on the next paragraph.
    """
    return apply_scene_markers(split_blocks(normalize_text(raw)))
