"""Domain models for vector storage (Phase 5.2)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class VectorRecord(BaseModel):
    """One dense vector with optional text and metadata payload."""

    id: str
    vector: list[float] = Field(min_length=1)
    text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorHit(BaseModel):
    """A similarity search hit from a vector store."""

    id: str
    score: float
    text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    vector: list[float] | None = None
