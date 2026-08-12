"""Shared enumerations used across agents and infrastructure."""

from __future__ import annotations

from enum import StrEnum


class QueryIntent(StrEnum):
    """High-level research intent derived from the user query."""

    FACTUAL = "factual"
    COMPARATIVE = "comparative"
    EXPLORATORY = "exploratory"
    HOWTO = "howto"
    NEWS = "news"
    ACADEMIC = "academic"
    CODE = "code"


class SearchProviderHint(StrEnum):
    """Preferred search backends for a research task (Phase 3.2)."""

    WEB = "web"
    GITHUB = "github"
    ARXIV = "arxiv"
    WIKIPEDIA = "wikipedia"
    YOUTUBE = "youtube"
    REDDIT = "reddit"


class ContentType(StrEnum):
    """Document content formats handled by Phase 4 parsers."""

    HTML = "html"
    MARKDOWN = "markdown"
    PDF = "pdf"
    OCR = "ocr"
    JSON = "json"


__all__ = [
    "ContentType",
    "QueryIntent",
    "SearchProviderHint",
]
