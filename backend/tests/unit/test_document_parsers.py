"""Tests for Phase 4.1 document parsers and DocumentParseService."""

from __future__ import annotations

from io import BytesIO

import pytest
from pypdf import PdfWriter

from insightforge.core.exceptions import ValidationFailedError
from insightforge.domain.models import Document
from insightforge.infrastructure.document import (
    DocumentParseService,
    HtmlDocumentParser,
    JsonDocumentParser,
    MarkdownDocumentParser,
    OcrDocumentParser,
    ParseRequest,
    PdfDocumentParser,
    build_parsers,
    create_document_parse_service,
    detect_content_type,
)
from insightforge.shared.enums import ContentType, SearchProviderHint


def _pdf_bytes() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = BytesIO()
    writer.write(buffer)
    payload = buffer.getvalue()
    assert payload.startswith(b"%PDF")
    return payload


def test_detect_content_type_from_mime_and_extension() -> None:
    assert detect_content_type(content_type="text/html; charset=utf-8") is ContentType.HTML
    assert detect_content_type(filename="notes.md") is ContentType.MARKDOWN
    assert detect_content_type(url="https://example.com/report.PDF") is ContentType.PDF
    assert detect_content_type(filename="scan.png") is ContentType.OCR
    assert detect_content_type(raw='{"title": "x"}') is ContentType.JSON
    assert detect_content_type(raw=b"%PDF-1.4") is ContentType.PDF


def test_html_parser_extracts_title_and_main_text() -> None:
    html = """
    <html>
      <head>
        <title>InsightForge</title>
        <meta name="description" content="Research platform" />
        <script>alert(1)</script>
      </head>
      <body>
        <nav>Skip me</nav>
        <main>
          <h1>Welcome</h1>
          <p>Normalized research sources.</p>
        </main>
      </body>
    </html>
    """
    parsed = HtmlDocumentParser().parse(html, source_url="https://example.com")

    assert parsed.content_type is ContentType.HTML
    assert parsed.title == "InsightForge"
    assert parsed.url == "https://example.com"
    assert "Welcome" in parsed.text
    assert "Normalized research sources." in parsed.text
    assert "alert(1)" not in parsed.text
    assert parsed.metadata["description"] == "Research platform"
    assert parsed.metadata["parser"] == "html"


def test_markdown_parser_frontmatter_and_heading() -> None:
    md = """---
title: Phase 4
author: Drashti
date: 2026-08-12
---

# Fallback Heading

Body paragraph.
"""
    parsed = MarkdownDocumentParser().parse(md, filename="phase4.md")

    assert parsed.content_type is ContentType.MARKDOWN
    assert parsed.title == "Phase 4"
    assert parsed.text.startswith("# Fallback Heading")
    assert parsed.metadata["author"] == "Drashti"
    assert parsed.metadata["date"] == "2026-08-12"
    assert parsed.metadata["filename"] == "phase4.md"


def test_markdown_parser_uses_heading_when_no_frontmatter_title() -> None:
    parsed = MarkdownDocumentParser().parse("# Only Heading\n\nText")
    assert parsed.title == "Only Heading"
    assert "Text" in parsed.text


def test_json_parser_prefers_content_field() -> None:
    raw = '{"title": "Doc", "content": "Hello JSON", "extra": 1}'
    parsed = JsonDocumentParser().parse(raw)

    assert parsed.content_type is ContentType.JSON
    assert parsed.title == "Doc"
    assert parsed.text == "Hello JSON"
    assert parsed.metadata["root_type"] == "dict"


def test_json_parser_flattens_without_content_field() -> None:
    parsed = JsonDocumentParser().parse('{"a": 1, "b": {"c": "x"}}')
    assert "a: 1" in parsed.text
    assert "b.c: x" in parsed.text


def test_json_parser_rejects_invalid_json() -> None:
    with pytest.raises(ValidationFailedError):
        JsonDocumentParser().parse("{not-json")


def test_pdf_parser_reads_valid_pdf() -> None:
    parsed = PdfDocumentParser().parse(_pdf_bytes(), filename="blank.pdf")

    assert parsed.content_type is ContentType.PDF
    assert parsed.metadata["page_count"] == 1
    assert parsed.metadata["parser"] == "pdf"
    assert parsed.metadata["filename"] == "blank.pdf"


def test_pdf_parser_rejects_corrupt_payload() -> None:
    with pytest.raises(ValidationFailedError):
        PdfDocumentParser().parse(b"not-a-pdf")


def test_ocr_parser_with_injected_engine() -> None:
    parser = OcrDocumentParser(extract_text=lambda _data: "OCR TEXT")
    assert parser.available

    # Minimal PNG header is enough because extract_text is injected.
    parsed = parser.parse(b"\x89PNG\r\n\x1a\nfake", filename="scan.png", title="Scan")

    assert parsed.content_type is ContentType.OCR
    assert parsed.text == "OCR TEXT"
    assert parsed.title == "Scan"
    assert parsed.metadata["engine"] == "custom"


def test_ocr_parser_unavailable_without_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = OcrDocumentParser(extract_text=None, tesseract_cmd="__missing_tesseract__")
    monkeypatch.setattr(
        "insightforge.infrastructure.document.parsers.ocr._tesseract_available",
        lambda _cmd=None: False,
    )
    assert not parser.available
    with pytest.raises(ValidationFailedError):
        parser.parse(b"\x89PNG\r\n\x1a\nfake")


def test_service_routes_by_filename() -> None:
    service = DocumentParseService(build_parsers(ocr_extract_text=lambda _: "x"))
    parsed = service.parse("# Hello", filename="readme.md", url="https://example.com/readme.md")

    assert parsed.content_type is ContentType.MARKDOWN
    assert parsed.title == "Hello"


def test_service_parse_source_uses_document_content() -> None:
    service = create_document_parse_service(ocr_extract_text=lambda _: "x")
    document = Document(
        title="Article",
        url="https://example.com/a.html",
        content="<html><head><title>X</title></head><body><p>Hi</p></body></html>",
        provider=SearchProviderHint.WEB,
        metadata={"content_type": "html"},
    )
    parsed = service.parse_source(document)

    assert parsed.content_type is ContentType.HTML
    assert parsed.title == "Article"
    assert "Hi" in parsed.text
    assert parsed.metadata["provider"] == "web"


def test_service_parse_many_soft_fails() -> None:
    service = DocumentParseService(build_parsers(ocr_extract_text=lambda _: "ok"))
    results = service.parse_many(
        [
            ParseRequest(raw="# Good", filename="a.md"),
            ParseRequest(raw="{bad", filename="b.json"),
            ParseRequest(raw='{"content": "fine"}', filename="c.json"),
        ]
    )

    assert len(results) == 2
    assert results[0].content_type is ContentType.MARKDOWN
    assert results[1].text == "fine"


def test_service_rejects_unknown_type() -> None:
    service = DocumentParseService(build_parsers())
    with pytest.raises(ValidationFailedError):
        service.parse("plain text with no hints")


def test_empty_payload_rejected() -> None:
    with pytest.raises(ValidationFailedError):
        HtmlDocumentParser().parse("   ")
