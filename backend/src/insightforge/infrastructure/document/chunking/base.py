"""Chunker contract and shared configuration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

from pydantic import BaseModel, Field, model_validator

from insightforge.core.exceptions import ValidationFailedError
from insightforge.domain.models import DocumentChunk, ParsedDocument
from insightforge.shared.enums import ChunkStrategy, ContentType

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_SEMANTIC_THRESHOLD = 0.25


class ChunkConfig(BaseModel):
    """Character-based chunking options (Phase 4.3)."""

    chunk_size: int = Field(default=DEFAULT_CHUNK_SIZE, ge=1)
    chunk_overlap: int = Field(default=DEFAULT_CHUNK_OVERLAP, ge=0)
    strategy: ChunkStrategy = ChunkStrategy.AUTO
    semantic_threshold: float = Field(default=DEFAULT_SEMANTIC_THRESHOLD, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_overlap(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValidationFailedError(
                "chunk_overlap must be smaller than chunk_size",
                details={
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                },
            )
        return self


class DocumentChunker(ABC):
    """One chunking strategy (recursive, markdown-aware, semantic)."""

    name: ChunkStrategy

    def __init__(self, config: ChunkConfig | None = None) -> None:
        self.config = config or ChunkConfig()

    @abstractmethod
    def chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        """Split ``document.text`` into ordered chunks."""


def parse_chunk_strategy(value: str | ChunkStrategy) -> ChunkStrategy:
    """Parse a strategy name from settings or callers."""

    if isinstance(value, ChunkStrategy):
        return value
    cleaned = value.strip().lower()
    try:
        return ChunkStrategy(cleaned)
    except ValueError as exc:
        raise ValidationFailedError(
            f"unknown chunk strategy {value!r}",
            details={
                "strategy": value,
                "allowed": [item.value for item in ChunkStrategy],
            },
        ) from exc


def resolve_strategy(
    document: ParsedDocument,
    requested: ChunkStrategy,
) -> ChunkStrategy:
    """``auto`` picks markdown-aware for Markdown, recursive otherwise."""

    if requested is not ChunkStrategy.AUTO:
        return requested
    if document.content_type is ContentType.MARKDOWN:
        return ChunkStrategy.MARKDOWN
    return ChunkStrategy.RECURSIVE
