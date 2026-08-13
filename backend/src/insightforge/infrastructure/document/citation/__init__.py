"""Phase 4.4 citation enrichment — URL, title, authors, date."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from insightforge.core.logging import get_logger
from insightforge.domain.models import Citation, DocumentChunk, ParsedDocument
from insightforge.infrastructure.document.citation.authors import parse_authors
from insightforge.infrastructure.document.citation.dates import normalize_date
from insightforge.infrastructure.document.citation.urls import clean_url, site_name

logger = get_logger(__name__)

__all__ = [
    "build_citation",
    "cite_chunks",
    "cite_document",
    "format_citation",
    "normalize_date",
    "parse_authors",
    "site_name",
]

_TITLE_KEYS = ("title", "headline", "name")
_DATE_KEYS = ("date", "published", "created", "publication_date", "published_at")
_AUTHOR_KEYS = ("authors", "author")


def format_citation(citation: Citation) -> str:
    """Render a compact citation string."""

    return citation.format()


def build_citation(
    *,
    title: str | None = None,
    url: str | None = None,
    metadata: dict[str, Any] | None = None,
    heading: str | None = None,
    locator: str | None = None,
) -> Citation:
    """Build a citation from explicit fields plus parser/cleaner metadata."""

    meta = metadata or {}
    resolved_url = clean_url(url) or clean_url(_as_optional_str(meta.get("url")))
    resolved_title = _first_str(title, *(_as_optional_str(meta.get(key)) for key in _TITLE_KEYS))
    if not resolved_title:
        resolved_title = _title_from_filename(_as_optional_str(meta.get("filename")))

    authors: list[str] = []
    for key in _AUTHOR_KEYS:
        authors = parse_authors(meta.get(key))
        if authors:
            break

    date_value = None
    for key in _DATE_KEYS:
        date_value = normalize_date(meta.get(key))
        if date_value:
            break

    citation = Citation(
        url=resolved_url,
        title=resolved_title,
        authors=authors,
        date=date_value,
        site_name=site_name(resolved_url),
        locator=locator or heading,
    )
    logger.debug(
        "citation built title=%s url=%s authors=%d date=%s locator=%s",
        bool(citation.title),
        bool(citation.url),
        len(citation.authors),
        citation.date,
        citation.locator,
    )
    return citation


def cite_document(document: ParsedDocument) -> ParsedDocument:
    """Attach a structured citation to a parsed document."""

    citation = build_citation(
        title=document.title,
        url=document.url,
        metadata=document.metadata,
    )
    logger.info(
        "document cited title=%s url=%s authors=%d date=%s",
        bool(citation.title),
        bool(citation.url),
        len(citation.authors),
        bool(citation.date),
        extra={
            "has_title": bool(citation.title),
            "has_url": bool(citation.url),
            "author_count": len(citation.authors),
            "has_date": bool(citation.date),
        },
    )
    return document.model_copy(
        update={"citation": citation, "title": citation.title or document.title}
    )


def cite_chunks(
    chunks: list[DocumentChunk],
    document: ParsedDocument | None = None,
) -> list[DocumentChunk]:
    """Attach per-chunk citations, inheriting source URL/title/authors/date."""

    base = document.citation if document is not None else None
    if base is None and document is not None:
        base = build_citation(
            title=document.title,
            url=document.url,
            metadata=document.metadata,
        )

    total = len(chunks)
    cited: list[DocumentChunk] = []
    for chunk in chunks:
        locator = chunk.heading or f"chunk {chunk.index + 1}/{total}"
        if base is not None:
            citation = base.model_copy(update={"locator": locator})
        else:
            citation = build_citation(
                title=chunk.title,
                url=chunk.url,
                metadata=chunk.metadata,
                heading=chunk.heading,
                locator=locator,
            )
        cited.append(chunk.model_copy(update={"citation": citation}))

    logger.info("chunks cited count=%d", len(cited), extra={"chunk_count": len(cited)})
    return cited


def _first_str(*values: str | None) -> str | None:
    for value in values:
        if value and value.strip():
            return value.strip()
    return None


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _title_from_filename(filename: str | None) -> str | None:
    if not filename:
        return None
    stem = Path(filename).stem.replace("_", " ").replace("-", " ").strip()
    return stem or None
