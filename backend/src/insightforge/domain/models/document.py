"""Domain document models for search hits and parsed source text."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, HttpUrl

from insightforge.shared.enums import ChunkStrategy, ContentType, SearchProviderHint


class Document(BaseModel):
    """A single retrieved source ready for ranking and report generation."""

    title: str
    url: str
    snippet: str = ""
    content: str = ""
    provider: SearchProviderHint
    score: float | None = Field(
        default=None,
        description=(
            "Application relevance score in 0..1 after search scoring. "
            "Provider-native signals stay in metadata and are not this field."
        ),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentRef(BaseModel):
    """Lightweight pointer used when only title/url are needed."""

    title: str
    url: HttpUrl | str
    provider: SearchProviderHint | None = None


class Citation(BaseModel):
    """Canonical bibliographic citation (Phase 4.4)."""

    url: str | None = None
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    date: str | None = None
    site_name: str | None = None
    locator: str | None = None

    def format(self) -> str:
        """Render a compact citation string for logs and reports."""

        pieces: list[str] = []
        authors = ", ".join(self.authors)
        if authors and self.date:
            pieces.append(f"{authors} ({self.date})")
        elif authors:
            pieces.append(authors)
        elif self.date:
            pieces.append(self.date)
        if self.title:
            pieces.append(self.title)
        if self.url:
            pieces.append(self.url)
        text = ". ".join(pieces)
        if self.locator:
            return f"{text} ({self.locator})" if text else self.locator
        return text


class ParsedDocument(BaseModel):
    """Text extracted from a raw source (Phase 4.1), optionally cleaned (4.2)."""

    text: str
    content_type: ContentType
    title: str | None = None
    url: str | None = None
    citation: Citation | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """A structured slice of a parsed document (Phase 4.3).

    ``start`` / ``end`` are character offsets into the source ``ParsedDocument.text``.
    """

    text: str
    index: int = Field(ge=0)
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    strategy: ChunkStrategy
    title: str | None = None
    url: str | None = None
    heading: str | None = None
    content_type: ContentType | None = None
    citation: Citation | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
