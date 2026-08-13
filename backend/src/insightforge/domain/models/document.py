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


class ParsedDocument(BaseModel):
    """Text extracted from a raw source (Phase 4.1) and optionally cleaned (4.2).

    Citation enrichment happens in Phase 4.4.
    """

    text: str
    content_type: ContentType
    title: str | None = None
    url: str | None = None
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
    metadata: dict[str, Any] = Field(default_factory=dict)
