"""Tests for Phase 4.2 document cleaning."""

from __future__ import annotations

from insightforge.domain.models import ParsedDocument
from insightforge.infrastructure.document import (
    DocumentParseService,
    build_parsers,
    clean_document,
    extract_text_metadata,
    normalize_text,
    remove_boilerplate,
)
from insightforge.shared.enums import ContentType


def test_normalize_text_collapses_whitespace_and_invisible_chars() -> None:
    raw = "Hello\u00ad  world\r\n\r\n\r\nNext\u200b line"
    assert normalize_text(raw) == "Hello world\n\nNext line"


def test_remove_boilerplate_drops_chrome_lines() -> None:
    text = "\n".join(
        [
            "Skip to main content",
            "Real research paragraph about transformers.",
            "Real research paragraph about transformers.",  # consecutive dup
            "Subscribe to our newsletter",
            "Another useful sentence.",
            "All rights reserved",
        ]
    )
    cleaned = remove_boilerplate(text, content_type=ContentType.HTML)

    assert "Skip to main content" not in cleaned
    assert "Subscribe to our newsletter" not in cleaned
    assert "All rights reserved" not in cleaned
    assert cleaned.count("Real research paragraph about transformers.") == 1
    assert "Another useful sentence." in cleaned


def test_remove_boilerplate_drops_markdown_share_links() -> None:
    text = "# Title\n\nBody text.\n\n[Share](https://example.com/share)\n\nMore body."
    cleaned = remove_boilerplate(text, content_type=ContentType.MARKDOWN)
    assert "[Share]" not in cleaned
    assert "Body text." in cleaned
    assert "More body." in cleaned


def test_extract_text_metadata_fills_missing_fields() -> None:
    text = "\n".join(
        [
            "By Jane Doe",
            "Published March 15, 2024",
            "See https://example.com/deep and contact jane@example.com",
            "Body continues here with enough words.",
        ]
    )
    meta = extract_text_metadata(text, existing={"parser": "html"}, title="Deep Dive")

    assert meta["author"] == "Jane Doe"
    assert meta["date"] == "March 15, 2024"
    assert meta["urls_in_text"] == ["https://example.com/deep"]
    assert meta["emails"] == ["jane@example.com"]
    assert meta["word_count"] > 0
    assert meta["cleaned"] is True
    assert meta["title"] == "Deep Dive"
    assert meta["parser"] == "html"


def test_extract_text_metadata_preserves_existing_author_and_date() -> None:
    meta = extract_text_metadata(
        "By Someone Else\nJanuary 1, 2020\nBody",
        existing={"author": "Keep Me", "date": "2026-01-01"},
    )
    assert meta["author"] == "Keep Me"
    assert meta["date"] == "2026-01-01"


def test_clean_document_pipeline() -> None:
    doc = ParsedDocument(
        text="Cookie policy\n\nHello\u00ad   world\n\n\n\nBy Ada Lovelace\n",
        content_type=ContentType.HTML,
        title="Note",
        url="https://example.com/note",
        metadata={"parser": "html"},
    )
    cleaned = clean_document(doc)

    assert "Cookie policy" not in cleaned.text
    assert "Hello world" in cleaned.text
    assert cleaned.metadata["author"] == "Ada Lovelace"
    assert cleaned.metadata["cleaned"] is True
    assert cleaned.metadata["chars_before_clean"] > cleaned.metadata["chars_after_clean"]


def test_service_applies_cleaning_by_default() -> None:
    service = DocumentParseService(build_parsers(ocr_extract_text=lambda _: "x"))
    parsed = service.parse(
        "# Hello\n\nAccept all cookies\n\nUseful content about LangGraph.",
        filename="readme.md",
    )

    assert parsed.content_type is ContentType.MARKDOWN
    assert "Accept all cookies" not in parsed.text
    assert "Useful content about LangGraph." in parsed.text
    assert parsed.metadata["cleaned"] is True


def test_service_can_skip_cleaning() -> None:
    service = DocumentParseService(
        build_parsers(ocr_extract_text=lambda _: "x"),
        cleaning=False,
    )
    parsed = service.parse(
        "# Hello\n\nAccept all cookies\n\nUseful content.",
        filename="readme.md",
    )
    assert "Accept all cookies" in parsed.text
    assert parsed.metadata.get("cleaned") is not True


def test_html_parser_strips_nav_footer_chrome() -> None:
    html = """
    <html><body>
      <main>
        <nav>Main menu</nav>
        <p>Core article text.</p>
        <footer>All rights reserved</footer>
      </main>
    </body></html>
    """
    service = DocumentParseService(build_parsers(ocr_extract_text=lambda _: "x"))
    parsed = service.parse(html, content_type="html")

    assert "Core article text." in parsed.text
    assert "Main menu" not in parsed.text
    assert "All rights reserved" not in parsed.text
