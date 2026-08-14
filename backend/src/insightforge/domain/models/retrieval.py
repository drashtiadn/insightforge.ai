"""Domain models for RAG retrieval (Phase 5.3)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from insightforge.shared.enums import RetrievalMode


class RetrievalHit(BaseModel):
    """One ranked chunk returned for a query."""

    id: str
    score: float
    text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    mode: RetrievalMode
    semantic_rank: int | None = Field(default=None, ge=1)
    bm25_rank: int | None = Field(default=None, ge=1)
    retrieval_score: float | None = None
    rerank_rank: int | None = Field(default=None, ge=1)
