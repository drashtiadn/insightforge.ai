"""Phase 4.3 document chunking — recursive, markdown-aware, semantic."""

from __future__ import annotations

from insightforge.core.logging import get_logger
from insightforge.domain.models import DocumentChunk, ParsedDocument
from insightforge.infrastructure.document.chunking.base import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEMANTIC_THRESHOLD,
    ChunkConfig,
    DocumentChunker,
    parse_chunk_strategy,
    resolve_strategy,
)
from insightforge.infrastructure.document.chunking.markdown import MarkdownAwareChunker
from insightforge.infrastructure.document.chunking.recursive import RecursiveCharacterChunker
from insightforge.infrastructure.document.chunking.semantic import EmbedFn, SemanticChunker
from insightforge.shared.enums import ChunkStrategy

logger = get_logger(__name__)

__all__ = [
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_SEMANTIC_THRESHOLD",
    "ChunkConfig",
    "DocumentChunker",
    "EmbedFn",
    "MarkdownAwareChunker",
    "RecursiveCharacterChunker",
    "SemanticChunker",
    "build_chunker",
    "chunk_document",
    "parse_chunk_strategy",
    "resolve_strategy",
]


def build_chunker(
    strategy: ChunkStrategy,
    config: ChunkConfig | None = None,
    *,
    embed_text: EmbedFn | None = None,
) -> DocumentChunker:
    """Construct a chunker for a resolved (non-auto) strategy."""

    cfg = config or ChunkConfig()
    if strategy is ChunkStrategy.MARKDOWN:
        return MarkdownAwareChunker(cfg)
    if strategy is ChunkStrategy.SEMANTIC:
        return SemanticChunker(cfg, embed_text=embed_text)
    return RecursiveCharacterChunker(cfg)


def chunk_document(
    document: ParsedDocument,
    *,
    config: ChunkConfig | None = None,
    strategy: ChunkStrategy | str | None = None,
    embed_text: EmbedFn | None = None,
) -> list[DocumentChunk]:
    """Split a parsed document using ``auto`` / recursive / markdown / semantic."""

    cfg = config or ChunkConfig()
    requested = parse_chunk_strategy(strategy) if strategy is not None else cfg.strategy
    resolved = resolve_strategy(document, requested)
    logger.info(
        "chunking started strategy=%s resolved=%s chars=%d size=%d overlap=%d",
        requested.value,
        resolved.value,
        len(document.text),
        cfg.chunk_size,
        cfg.chunk_overlap,
        extra={"strategy": requested.value, "resolved": resolved.value},
    )
    chunks = build_chunker(resolved, cfg, embed_text=embed_text).chunk(document)
    logger.info(
        "chunking finished strategy=%s chunks=%d",
        resolved.value,
        len(chunks),
        extra={"strategy": resolved.value, "chunk_count": len(chunks)},
    )
    return chunks
