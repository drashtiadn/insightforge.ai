"""Embeddings infrastructure — provider abstraction for dense text vectors."""

from insightforge.infrastructure.embeddings.base import EmbeddingProvider
from insightforge.infrastructure.embeddings.http import (
    batched,
    close_http_client,
    create_http_client,
    require_texts,
)
from insightforge.infrastructure.embeddings.providers import (
    LocalEmbeddingProvider,
    LocalEmbedFn,
    VoyageEmbeddingProvider,
)
from insightforge.infrastructure.embeddings.service import (
    EmbeddingService,
    build_provider,
    create_embedding_service,
    parse_embedding_provider,
)

__all__ = [
    "EmbeddingProvider",
    "EmbeddingService",
    "LocalEmbedFn",
    "LocalEmbeddingProvider",
    "VoyageEmbeddingProvider",
    "batched",
    "build_provider",
    "close_http_client",
    "create_embedding_service",
    "create_http_client",
    "parse_embedding_provider",
    "require_texts",
]
