"""Ingest: loaders (txt/md/epub/pdf), chapter detection and Book assembly."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
from ebooklib import epub

from bookreader.ingest import load_book
from bookreader.ingest.chapters import detect_chapters, heading_title, merge_short_chapters
from bookreader.ingest.epub import epub_to_sections, extract_blocks
from bookreader.ingest.pdf import unwrap_lines
from bookreader.ingest.text import text_to_paragraphs
from bookreader.types import InputError

FIXTURE_TITLE = "The Lighthouse at Gull Point"
FIXTURE_CHAPTERS = ["Chapter 1: The Storm Bell", "Chapter 2: The Stranger", "Chapter 3: The Letter"]

_WORDS = ("the sea rolled grey and slow beneath a sky the colour of old pewter while "
          "gulls wheeled above the point and the keeper watched the horizon for sails").split()


def prose(n_words: int, seed: int = 0) -> str:
    """Deterministic lowercase prose of *n_words* words that never looks like a heading."""
    words = [_WORDS[(i * 5 + seed) % len(_WORDS)] for i in range(n_words)]
    sentences = [" ".join(words[i:i + 12]).capitalize() + "." for i in range(0, len(words), 12)]
    return " ".join(sentences)


def paragraphs_of(n_words: int, per_paragraph: int = 60, seed: int = 0) -> list[str]:
    return [prose(per_paragraph, seed + i) for i in range(max(1, n_words // per_paragraph))]


# --------------------------------------------------------------------------- fixture book
def test_fixture_title_and_chapters(sample_book_path: Path) -> None:
    book = load_book(sample_book_path)
    assert book.title == FIXTURE_TITLE
    assert [c.title for c in book.chapters] == FIXTURE_CHAPTERS
    assert [c.index for c in book.chapters] == [1, 2, 3]
    assert len(book.chapters[0].paragraphs) == 9
    assert book.chapters[0].paragraphs[0].text.startswith("The wind came off the sea")
    assert book.source_filename == "sample_book.txt"


def test_fixture_sha256_and_word_count(sample_book_path: Path) -> None:
    book = load_book(sample_book_path)
    assert book.source_sha256 == hashlib.sha256(sample_book_path.read_bytes()).hexdigest()
    assert book.source_sha256 == load_book(sample_book_path).source_sha256
    assert book.word_count > 500
    assert book.word_count == sum(len(p.text.split()) for c in book.chapters for p in c.paragraphs)
    assert book.quote_style == "double"


def test_fixture_headings_are_not_paragraphs_or_spans(sample_book_path: Path) -> None:
    book = load_book(sample_book_path)
    texts = [p.text for c in book.chapters for p in c.paragraphs]
    assert FIXTURE_TITLE not in texts
    assert not any(t.startswith("Chapter") for t in texts)
    span_ids = [s.id for c in book.chapters for s in c.spans]
    assert len(span_ids) == len(set(span_ids))
    assert all(p.spans for c in book.chapters for p in c.paragraphs)
    assert [p.index for p in book.chapters[1].paragraphs] == list(range(1, 16))
    assert book.chapters[0].paragraphs[1].spans[0].id == "c1p2s0"


def test_title_hint_used_only_without_detected_title(sample_book_path: Path, tmp_path: Path) -> None:
    assert load_book(sample_book_path, title_hint="Ignored").title == FIXTURE_TITLE
    path = tmp_path / "plain_story.txt"
    path.write_text("\n\n".join(paragraphs_of(400)), encoding="utf-8")
    assert load_book(path, title_hint="From The Form").title == "From The Form"
    assert load_book(path).title == "plain_story"


# --------------------------------------------------------------------------- text normalization
def test_crlf_input_matches_lf_input(sample_book_path: Path, tmp_path: Path) -> None:
    crlf = tmp_path / "crlf.txt"
    crlf.write_bytes(sample_book_path.read_bytes().replace(b"\n", b"\r\n"))
    lf_book = load_book(sample_book_path)
    crlf_book = load_book(crlf)
    assert crlf_book.title == FIXTURE_TITLE
    assert [c.title for c in crlf_book.chapters] == FIXTURE_CHAPTERS
    assert [p.text for c in crlf_book.chapters for p in c.paragraphs] == [p.text for c in lf_book.chapters for p in c.paragraphs]
    assert "\r" not in crlf_book.chapters[0].text
    assert crlf_book.source_sha256 != lf_book.source_sha256


def test_markdown_headings(tmp_path: Path) -> None:
    body = "\n\n".join(paragraphs_of(200))
    path = tmp_path / "book.md"
    path.write_text(f"# The Book\n\n## Chapter 1\n\n{body}\n\n## Chapter 2\n{body}\n", encoding="utf-8")
    book = load_book(path)
    assert book.title == "The Book"
    assert [c.title for c in book.chapters] == ["Chapter 1", "Chapter 2"]
    assert all(not p.text.startswith("#") for c in book.chapters for p in c.paragraphs)
    assert len(book.chapters[1].paragraphs) == len(book.chapters[0].paragraphs)


def test_scene_breaks_from_markers_and_blank_lines() -> None:
    raw = "One.\n\nTwo.\n\n* * *\n\nThree.\n\n\n\nFour.\n\n---\n\nFive.\r\nstill five.\n"
    assert text_to_paragraphs(raw) == [
        ("One.", False), ("Two.", False), ("Three.", True), ("Four.", True), ("Five. still five.", True),
    ]


def test_scene_break_flag_survives_chapter_assembly(tmp_path: Path) -> None:
    paras = paragraphs_of(300)
    text = "Title\n\nChapter 1\n\n" + "\n\n".join(paras[:2]) + "\n\n***\n\n" + "\n\n".join(paras[2:]) + "\n\nChapter 2\n\n" + "\n\n".join(paras)
    path = tmp_path / "b.txt"
    path.write_text(text, encoding="utf-8")
    book = load_book(path)
    flags = [p.scene_break_before for p in book.chapters[0].paragraphs]
    assert flags == [False, False, True] + [False] * (len(paras) - 3)
    assert not any(p.scene_break_before for p in book.chapters[1].paragraphs)


def test_utf8_bom_and_latin1_fallback(tmp_path: Path) -> None:
    body = "\n\n".join(paragraphs_of(200))
    bom = tmp_path / "bom.txt"
    bom.write_bytes(b"\xef\xbb\xbfTitle\n\nChapter 1\n\n" + body.encode() + b"\n\nChapter 2\n\n" + body.encode())
    assert load_book(bom).title == "Title"
    latin = tmp_path / "latin.txt"
    latin.write_bytes(("Café Title\n\nChapter 1\n\n" + body + "\n\nChapter 2\n\n" + body).encode("cp1252"))
    assert load_book(latin).title == "Café Title"


# --------------------------------------------------------------------------- chapter detection
def test_heading_heuristics() -> None:
    assert heading_title("Chapter 1: The Storm Bell") == "Chapter 1: The Storm Bell"
    assert heading_title("PART TWO") == "PART TWO"
    assert heading_title("prologue") == "prologue"
    assert heading_title("## The Stranger ##") == "The Stranger"
    assert heading_title("THE STORM BELL") == "THE STORM BELL"
    assert heading_title("XIV") == "XIV"
    assert heading_title("12.") == "12."
    assert heading_title("The wind came off the sea like a living thing.") is None
    assert heading_title("NO") is None                         # one word
    assert heading_title('"GET OUT!" she screamed.') is None    # dialogue
    assert heading_title("Chapter " + "x" * 80) is None         # too long
    assert heading_title("A B C D E F G H I") is None           # nine words


def test_keyword_initial_sentences_are_not_headings() -> None:
    for sentence in (
        "Part of her wanted to run.", "Book me a room at the inn, she thought.", "Interlude music drifted up from below.",
        "Chapter and verse, he quoted.", "Part civil war", "I SAID NO.", "'KEEP OUT'", "TO WHOM IT MAY CONCERN,",
    ):
        assert heading_title(sentence) is None, sentence
    for heading in (
        "Part One", "PART TWO", "Book II: The Return", "Chapter 1: The Storm Bell", "Prologue", "Epilogue - Ten Years Later",
        "Chapter 3.", "Chapter Twenty-One", "Chapter 12 The Storm", "Interlude: Winter", "THE STORM BELL", "KEEP OUT BY ORDER",
    ):
        assert heading_title(heading) == heading, heading


def test_keyword_initial_sentence_inside_a_chapter_does_not_split_it() -> None:
    body = paragraphs_of(200)
    sentence = "Part of her wanted to run."
    title, chapters = detect_chapters(["The Roof", "Chapter 1", *body[:2], sentence, *body[2:], "Chapter 2", *body])
    assert title == "The Roof"
    assert [t for t, _ in chapters] == ["Chapter 1", "Chapter 2"]
    assert chapters[0][1] == [*body[:2], sentence, *body[2:]], "the sentence is spoken, not turned into a title"


def test_pre_heading_prose_is_kept_not_dropped() -> None:
    body = paragraphs_of(200)
    opening = prose(50, seed=9)
    title, chapters = detect_chapters(["The Roof", opening, "Chapter 1", *body, "Chapter 2", *body])
    assert title == "The Roof"
    assert [t for t, _ in chapters] == ["Chapter 1", "Chapter 2"], "a short prologue merges into the first chapter"
    assert chapters[0][1] == [opening, *body]
    # a book with no real headings and two keyword-initial sentences keeps every paragraph
    title, chapters = detect_chapters(["The Roof", opening, "Part of her wanted to run.", prose(60, seed=3), "Book me a room, she thought.", prose(80, seed=4)])
    assert title is None and [t for t, _ in chapters] == ["Part 1"]
    assert len(chapters[0][1]) == 6


def test_all_caps_lines_inside_a_keyworded_book_stay_spoken() -> None:
    body = paragraphs_of(200)
    signs = ["KEEP OUT BY ORDER", "THIS MEANS YOU"]
    title, chapters = detect_chapters(["The Gate", "Chapter 1", *body[:1], *signs, *body[1:], "Chapter 2", *body, "THE END"])
    assert [t for t, _ in chapters] == ["Chapter 1", "Chapter 2"]
    assert chapters[0][1] == [*body[:1], *signs, *body[1:]], "the sign lines are paragraphs, not chapter titles"
    assert "THE END" not in chapters[1][1], "a trailing caps line is still a trailer"
    # a caps subtitle right after a keyworded heading is still a heading (an empty chapter that merges away)
    title, chapters = detect_chapters(["The Gate", "Chapter 1", "THE STORM BELL", *body, "Chapter 2", *body])
    assert [t for t, _ in chapters] == ["THE STORM BELL", "Chapter 2"]
    assert "THE STORM BELL" not in chapters[0][1]
    # books whose only headings are caps lines keep working
    title, chapters = detect_chapters(["THE WATCH", "THE FIRST NIGHT", *body, "THE SECOND NIGHT", *body])
    assert title == "THE WATCH" and [t for t, _ in chapters] == ["THE FIRST NIGHT", "THE SECOND NIGHT"]


def test_headingless_text_gets_synthetic_parts(tmp_path: Path) -> None:
    paras = paragraphs_of(15_000, per_paragraph=75)
    path = tmp_path / "long.txt"
    path.write_text("\n\n".join(paras), encoding="utf-8")
    book = load_book(path, title_hint="Long Book")
    assert book.title == "Long Book"
    assert [c.title for c in book.chapters] == ["Part 1", "Part 2", "Part 3"]
    assert [p.text for c in book.chapters for p in c.paragraphs] == paras
    sizes = [sum(len(p.text.split()) for p in c.paragraphs) for c in book.chapters]
    assert all(4000 <= s <= 6000 for s in sizes)
    assert book.word_count == 15_000


def test_detect_chapters_prologue_and_short_chapter_merge() -> None:
    intro = paragraphs_of(150)
    body = paragraphs_of(200)
    paragraphs = ["My Title", "by Someone", *intro, "Chapter 1", "Only a few words here.", "Chapter 2", *body, "THE END"]
    title, chapters = detect_chapters(paragraphs)
    assert title == "My Title"
    assert [t for t, _ in chapters] == ["Prologue", "Chapter 2"]
    assert chapters[0][1] == ["by Someone", *intro]
    assert chapters[1][1] == ["Only a few words here.", *body]


def test_detect_chapters_drops_short_front_matter_and_needs_two_headings() -> None:
    body = paragraphs_of(200)
    title, chapters = detect_chapters(["The Title", "by Someone", "Chapter 1", *body, "Chapter 2", *body])
    assert title == "The Title"
    assert [t for t, _ in chapters] == ["Chapter 1", "Chapter 2"]
    assert chapters[0][1] == body
    title, chapters = detect_chapters(["Chapter 1", *body])
    assert title is None
    assert chapters == [("Part 1", ["Chapter 1", *body])]


def test_merge_short_chapters_at_end() -> None:
    body = paragraphs_of(200)
    merged = merge_short_chapters([("One", body), ("Two", ["tiny"]), ("Three", ["last"])])
    assert merged == [("One", body + ["tiny", "last"])]
    assert merge_short_chapters([("Only", ["tiny"])]) == [("Only", ["tiny"])]


# --------------------------------------------------------------------------- epub
def _write_epub(path: Path, *, headings: bool = True, title: str | None = "An EPUB Story") -> None:
    book = epub.EpubBook()
    book.set_identifier("test-epub-1")
    if title:
        book.set_title(title)
    book.set_language("en")
    front = epub.EpubHtml(title="Contents", file_name="front.xhtml", lang="en")
    front.content = "<html><body><nav><ol><li>" + " ".join(["word"] * 20) + "</li></ol></nav></body></html>"
    items = [front]
    if headings:
        contents = [
            f"<h1>Chapter One</h1><p>{prose(160, 1)}</p><p>{prose(80, 2)}</p>",
            f"<div><h2>Chapter Two</h2><p>{prose(160, 3)}</p><hr/><p>{prose(80, 4)}</p>"
            f"<h2>Chapter Three</h2><p>{prose(160, 5)}</p></div>",
            f"<h1>Chapter <em>Four</em></h1><p>{prose(120, 6)}<br/>{prose(60, 7)}</p><blockquote>{prose(30, 8)}</blockquote>",
        ]
    else:
        contents = [f"<p>{prose(200, i)}</p><p>{prose(200, i + 1)}</p>" for i in range(3)]
    for n, content in enumerate(contents, start=1):
        item = epub.EpubHtml(title=f"Item {n}", file_name=f"item{n}.xhtml", lang="en")
        item.content = f"<html><head><title>Item {n}</title></head><body>{content}</body></html>"
        items.append(item)
    for item in items:
        book.add_item(item)
    book.toc = tuple(items[1:])
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", *items]
    epub.write_epub(str(path), book)


def test_epub_sections_and_chapters(tmp_path: Path) -> None:
    path = tmp_path / "story.epub"
    _write_epub(path)
    title, sections = epub_to_sections(path)
    assert title == "An EPUB Story"
    assert [t for t, _ in sections] == ["Chapter One", "Chapter Two", "Chapter Three", "Chapter Four"]
    assert not any("word word" in p for _, paras in sections for p in paras)

    book = load_book(path)
    assert book.title == "An EPUB Story"
    assert [c.title for c in book.chapters] == ["Chapter One", "Chapter Two", "Chapter Three", "Chapter Four"]
    assert len(book.chapters[0].paragraphs) == 2
    assert [p.scene_break_before for p in book.chapters[1].paragraphs] == [False, True]
    assert len(book.chapters[3].paragraphs) == 2
    assert "Item" not in book.chapters[0].text
    assert book.word_count > 800


def test_epub_without_headings_falls_through_to_detect_chapters(tmp_path: Path) -> None:
    path = tmp_path / "flat.epub"
    _write_epub(path, headings=False, title=None)
    book = load_book(path, title_hint="Hinted")
    assert book.title == "Hinted"
    assert [c.title for c in book.chapters] == ["Part 1"]
    assert len(book.chapters[0].paragraphs) == 6


def test_epub_corrupt_file_raises_input_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.epub"
    path.write_bytes(b"not a zip file at all")
    with pytest.raises(InputError):
        load_book(path)


def test_extract_blocks_handles_nesting_and_entities() -> None:
    html = ("<body><h1>Title &amp; More</h1><div><p>First <b>bold</b> para.</p>"
            "<p>Second&nbsp;para</p></div><h3>Sub</h3><ul><li>one</li><li>two</li></ul>"
            "<script>ignored()</script><p>  </p></body>")
    assert extract_blocks(html) == [
        ("heading", "Title & More"), ("text", "First bold para."), ("text", "Second para"),
        ("text", "Sub"), ("text", "one"), ("text", "two"),
    ]


# --------------------------------------------------------------------------- pdf
def test_pdf_without_pypdf_raises_actionable_input_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pypdf", None)
    path = tmp_path / "book.pdf"
    path.write_bytes(b"%PDF-1.4\n%%EOF\n")
    with pytest.raises(InputError, match=r"bookreader\[pdf\]"):
        load_book(path)


def test_unwrap_lines_dehyphenates_and_joins() -> None:
    raw = ("The keeper climbed the light-\nhouse stairs two at a time, lantern swing-\ning wildly.\n"
           "It was late.\nMorning came grey and quiet and the storm had blown itself out over the\nwater.\n\n"
           "Chapter 2\nMore text follows here.")
    out = unwrap_lines(raw)
    assert "lighthouse" in out and "swinging" in out
    assert "time, lantern swinging wildly. It was late." in out
    assert "\n\nMorning" in out
    assert "over the water." in out
    assert out.endswith("Chapter 2\n\nMore text follows here.")


def _minimal_pdf(pages: list[list[str]]) -> bytes:
    """Hand-built PDF: one Helvetica text object per line, enough for pypdf.extract_text()."""
    objects: list[bytes] = []
    n_pages = len(pages)
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n_pages))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    font_id = 3 + 2 * n_pages
    for i, lines in enumerate(pages):
        content = b"BT /F1 12 Tf 72 720 Td 14 TL " + b" ".join(
            b"(" + line.encode("latin-1").replace(b"(", b"\\(").replace(b")", b"\\)") + b") Tj T*" for line in lines
        ) + b" ET"
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 {font_id} 0 R >> >> "
            f"/Contents {4 + 2 * i} 0 R >>".encode()
        )
        objects.append(b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for num, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{num} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def test_pdf_round_trip_with_pypdf(tmp_path: Path) -> None:
    pytest.importorskip("pypdf")
    body = paragraphs_of(300, per_paragraph=50)
    lines_1 = ["A PDF Title", "", "Chapter 1", *(body[0].split(". ")), "1"]
    lines_2 = ["2", *(body[1].split(". ")), "", "Chapter 2", *(body[2].split(". ")), *(body[3].split(". "))]
    path = tmp_path / "book.pdf"
    path.write_bytes(_minimal_pdf([lines_1, lines_2, [*(body[4].split(". ")), *(body[5].split(". "))]]))
    book = load_book(path)
    assert book.title == "A PDF Title"
    assert [c.title for c in book.chapters] == ["Chapter 1", "Chapter 2"]
    assert book.word_count >= 250
    assert not any(p.text.strip() in {"1", "2"} for c in book.chapters for p in c.paragraphs)


# --------------------------------------------------------------------------- guards
def test_unsupported_suffix_raises_input_error(tmp_path: Path) -> None:
    path = tmp_path / "book.exe"
    path.write_bytes(b"MZ" + b"\0" * 100)
    with pytest.raises(InputError, match="unsupported"):
        load_book(path)


def test_too_short_book_raises_input_error(tmp_path: Path) -> None:
    path = tmp_path / "short.txt"
    path.write_text("Just a handful of words, nowhere near a book.", encoding="utf-8")
    with pytest.raises(InputError, match="50"):
        load_book(path)


def test_missing_file_raises_input_error(tmp_path: Path) -> None:
    with pytest.raises(InputError):
        load_book(tmp_path / "nope.txt")
