"""Tests for Phase 4.4 citation enrichment."""

from __future__ import annotations

from datetime import date, datetime

from insightforge.domain.models import Citation, ParsedDocument
from insightforge.infrastructure.document import (
    ChunkConfig,
    DocumentParseService,
    build_citation,
    build_parsers,
    cite_document,
    format_citation,
    normalize_date,
    parse_authors,
)
from insightforge.shared.enums import ContentType


def test_parse_authors_splits_lists_and_and_clauses() -> None:
    assert parse_authors("Jane Doe and John Smith") == ["Jane Doe", "John Smith"]
    assert parse_authors("Ada Lovelace; Grace Hopper") == ["Ada Lovelace", "Grace Hopper"]
    assert parse_authors(["Ada Lovelace", "Grace Hopper"]) == ["Ada Lovelace", "Grace Hopper"]
    assert parse_authors("Doe, Jane") == ["Doe, Jane"]
    assert parse_authors("Jane Doe, John Smith") == ["Jane Doe", "John Smith"]


def test_normalize_date_iso_and_natural() -> None:
    assert normalize_date("2024-03-15") == "2024-03-15"
    assert normalize_date("March 15, 2024") == "2024-03-15"
    assert normalize_date("15 March 2024") == "2024-03-15"
    assert normalize_date("D:20240315120000+00'00'") == "2024-03-15"
    assert normalize_date(date(2024, 3, 15)) == "2024-03-15"
    assert normalize_date(datetime(2024, 3, 15, 12, 0)) == "2024-03-15"
    assert normalize_date("Spring 2024") == "Spring 2024"


def test_build_citation_from_metadata_and_url() -> None:
    citation = build_citation(
        title="InsightForge",
        url="https://www.example.com/research?utm_source=x",
        metadata={"author": "Ada Lovelace and Grace Hopper", "date": "March 15, 2024"},
    )
    assert citation.title == "InsightForge"
    assert citation.url == "https://www.example.com/research?utm_source=x"
    assert citation.site_name == "example.com"
    assert citation.authors == ["Ada Lovelace", "Grace Hopper"]
    assert citation.date == "2024-03-15"
    assert "Ada Lovelace, Grace Hopper (2024-03-15)" in format_citation(citation)
    assert "InsightForge" in citation.format()


def test_build_citation_falls_back_to_filename_title() -> None:
    citation = build_citation(metadata={"filename": "phase-4-notes.md"})
    assert citation.title == "phase 4 notes"


def test_cite_document_attaches_citation() -> None:
    parsed = ParsedDocument(
        text="Body",
        content_type=ContentType.MARKDOWN,
        title="Phase 4",
        url="https://example.com/a",
        metadata={"author": "Drashti", "date": "2026-08-12", "parser": "markdown"},
    )
    cited = cite_document(parsed)
    assert cited.citation is not None
    assert cited.citation.authors == ["Drashti"]
    assert cited.citation.date == "2026-08-12"
    assert cited.citation.url == "https://example.com/a"


def test_service_parse_builds_citation_from_frontmatter() -> None:
    service = DocumentParseService(build_parsers(ocr_extract_text=lambda _: "x"))
    parsed = service.parse(
        """---
title: Phase 4
author: Drashti Adani
date: 2026-08-12
---

# Fallback

Body paragraph about citation enrichment.
""",
        filename="phase4.md",
        url="https://example.com/phase4.md",
    )
    assert parsed.citation is not None
    assert parsed.citation.title == "Phase 4"
    assert parsed.citation.authors == ["Drashti Adani"]
    assert parsed.citation.date == "2026-08-12"
    assert parsed.citation.url == "https://example.com/phase4.md"
    assert parsed.citation.site_name == "example.com"


def test_service_parse_html_meta_author_and_date() -> None:
    html = """
    <html>
      <head>
        <title>Research Notes</title>
        <meta name="author" content="Jane Doe" />
        <meta property="article:published_time" content="2024-03-15T10:00:00Z" />
      </head>
      <body><main><p>Useful article text about citations.</p></main></body>
    </html>
    """
    service = DocumentParseService(build_parsers(ocr_extract_text=lambda _: "x"))
    parsed = service.parse(html, content_type="html", url="https://example.com/notes")
    assert parsed.citation is not None
    assert parsed.citation.title == "Research Notes"
    assert parsed.citation.authors == ["Jane Doe"]
    assert parsed.citation.date == "2024-03-15"


def test_service_parse_json_authors_list() -> None:
    raw = (
        '{"title": "JSON Doc", "authors": ["Ada Lovelace", "Grace Hopper"],'
        ' "date": "15 March 2024", "content": "Hello JSON citation."}'
    )
    service = DocumentParseService(build_parsers(ocr_extract_text=lambda _: "x"))
    parsed = service.parse(raw, filename="doc.json", url="https://example.com/doc.json")
    assert parsed.citation is not None
    assert parsed.citation.authors == ["Ada Lovelace", "Grace Hopper"]
    assert parsed.citation.date == "2024-03-15"


def test_service_parse_and_chunk_attaches_locator() -> None:
    service = DocumentParseService(
        build_parsers(ocr_extract_text=lambda _: "x"),
        chunk_config=ChunkConfig(chunk_size=80, chunk_overlap=10),
    )
    chunks = service.parse_and_chunk(
        """---
title: Notes
author: Ada Lovelace
date: 2024-03-15
---

# Intro

Opening paragraph about citation locators.

## Setup

Install the package and configure keys for research.
""",
        filename="notes.md",
        url="https://example.com/notes.md",
    )
    assert chunks
    assert all(chunk.citation is not None for chunk in chunks)
    first = chunks[0].citation
    assert first is not None
    assert first.authors == ["Ada Lovelace"]
    assert first.date == "2024-03-15"
    assert first.url == "https://example.com/notes.md"
    assert first.locator
    assert any(
        chunk.citation is not None and "Setup" in (chunk.citation.locator or "") for chunk in chunks
    )


def test_service_can_skip_citation() -> None:
    service = DocumentParseService(
        build_parsers(ocr_extract_text=lambda _: "x"),
        citation=False,
    )
    parsed = service.parse("# Hello\n\nBody", filename="a.md")
    assert parsed.citation is None


def test_citation_format_handles_partial_fields() -> None:
    assert Citation(url="https://example.com").format() == "https://example.com"
    assert Citation(title="Only Title").format() == "Only Title"
    assert "chunk 1/3" in Citation(title="Doc", locator="chunk 1/3").format()
