"""Domain document model returned by search providers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, HttpUrl

from insightforge.shared.enums import SearchProviderHint


class Document(BaseModel):
    """A single retrieved source ready for ranking and report generation."""

    title: str
    url: str
    snippet: str = ""
    content: str = ""
    provider: SearchProviderHint
    score: float | None = Field(
        default=None,
        description="Optional relevance score; filled by Phase 3.3 scoring",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentRef(BaseModel):
    """Lightweight pointer used when only title/url are needed."""

    title: str
    url: HttpUrl | str
    provider: SearchProviderHint | None = None
