"""Domain models for text embeddings (Phase 5.1)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from insightforge.shared.enums import EmbeddingInputType, EmbeddingProviderHint


class EmbeddingResult(BaseModel):
    """Vectors produced for one embed call."""

    vectors: list[list[float]]
    model: str
    provider: EmbeddingProviderHint
    dimensions: int = Field(ge=0)
    input_type: EmbeddingInputType | None = None
    usage_tokens: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
