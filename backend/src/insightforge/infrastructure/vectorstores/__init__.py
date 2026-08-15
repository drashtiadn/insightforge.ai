"""Vector store infrastructure — dense indexes for RAG retrieval."""

from insightforge.infrastructure.vectorstores.base import VectorStore
from insightforge.infrastructure.vectorstores.helpers import (
    cosine_similarity,
    matches_filters,
    require_records,
    require_vector,
)
from insightforge.infrastructure.vectorstores.service import (
    VectorStoreService,
    build_store,
    create_vector_store_service,
    parse_vector_store,
)
from insightforge.infrastructure.vectorstores.session import SessionVectorStoreRegistry
from insightforge.infrastructure.vectorstores.stores import (
    MemoryVectorStore,
    QdrantVectorStore,
)

__all__ = [
    "MemoryVectorStore",
    "QdrantVectorStore",
    "SessionVectorStoreRegistry",
    "VectorStore",
    "VectorStoreService",
    "build_store",
    "cosine_similarity",
    "create_vector_store_service",
    "matches_filters",
    "parse_vector_store",
    "require_records",
    "require_vector",
]
