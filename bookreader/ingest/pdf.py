"""bookreader.ingest.pdf - PDF text extraction (optional ``pypdf`` dependency)."""
from __future__ import annotations

import logging
import re
from pathlib import Path

from bookreader.ingest.chapters import heading_title
from bookreader.types import InputError

log = logging.getLogger(__name__)

PDF_EXTRA_HINT = "PDF support needs: pip install 'bookreader[pdf]'"
_SHORT_LINE_RATIO = 0.6
_TERMINAL_PUNCTUATION = ".!?\"'”’»:"
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(?=[a-z])")
_PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s+)?\d{1,4}\s*$", re.IGNORECASE)


def _page_lines(text: str) -> list[str]:
    """Lines of one page with page-number lines at the top/bottom removed."""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and (not lines[0].strip() or _PAGE_NUMBER_RE.match(lines[0])):
        lines.pop(0)
    while lines and (not lines[-1].strip() or _PAGE_NUMBER_RE.match(lines[-1])):
        lines.pop()
    return lines


def unwrap_lines(text: str) -> str:
    """De-hyphenate line ends and join hard-wrapped lines into paragraphs.

    A line ends a paragraph when it is blank, when it ends with terminal punctuation and is
    noticeably shorter than the page's typical line, or when it (or the next line) looks like a
    chapter heading; every other line break becomes a space.
    """
    text = _HYPHEN_BREAK_RE.sub(r"\1", text)
    lines = text.split("\n")
    lengths = sorted(len(line) for line in lines if line.strip())
    typical = lengths[len(lengths) // 2] if lengths else 0
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        heading = heading_title(stripped) is not None
        if heading and current:
            paragraphs.append(" ".join(current))
            current = []
        current.append(stripped)
        short = stripped[-1] in _TERMINAL_PUNCTUATION and len(stripped) < typical * _SHORT_LINE_RATIO
        if heading or short:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def pdf_to_text(path: Path) -> str:
    """Extract the text of a PDF as paragraphs separated by blank lines.

    ``pypdf`` is imported lazily; when it is missing an :class:`InputError` naming the pip
    extra is raised so the job fails with an actionable message instead of a crash.
    """
    try:
        import pypdf
    except ImportError as exc:
        raise InputError(PDF_EXTRA_HINT) from exc
    try:
        reader = pypdf.PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # pypdf raises several error types for damaged files
        raise InputError(f"cannot read PDF {path.name}: {exc}") from exc
    log.info("pdf %s: %d page(s)", path.name, len(pages))
    joined = "\n".join("\n".join(_page_lines(page)) for page in pages)
    return unwrap_lines(joined)
