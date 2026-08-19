"""Shared enumerations used across agents and infrastructure."""

from __future__ import annotations

from enum import StrEnum


class QueryIntent(StrEnum):
    """High-level research intent derived from the user query."""

    FACTUAL = "factual"
    COMPARATIVE = "comparative"
    EXPLORATORY = "exploratory"
    HOWTO = "howto"
    NEWS = "news"
    ACADEMIC = "academic"
    CODE = "code"


class SearchProviderHint(StrEnum):
    """Preferred search backends for a research task (Phase 3.2)."""

    WEB = "web"
    GITHUB = "github"
    ARXIV = "arxiv"
    WIKIPEDIA = "wikipedia"
    YOUTUBE = "youtube"
    REDDIT = "reddit"


class ContentType(StrEnum):
    """Document content formats handled by Phase 4 parsers."""

    HTML = "html"
    MARKDOWN = "markdown"
    PDF = "pdf"
    OCR = "ocr"
    JSON = "json"


class ChunkStrategy(StrEnum):
    """Chunking strategies for Phase 4.3."""

    AUTO = "auto"
    RECURSIVE = "recursive"
    MARKDOWN = "markdown"
    SEMANTIC = "semantic"


class EmbeddingProviderHint(StrEnum):
    """Embedding backends for Phase 5.1."""

    VOYAGE = "voyage"
    LOCAL = "local"


class EmbeddingInputType(StrEnum):
    """Voyage-style hint for asymmetric retrieval embeddings."""

    QUERY = "query"
    DOCUMENT = "document"


class VectorStoreHint(StrEnum):
    """Vector store backends for Phase 5.2."""

    QDRANT = "qdrant"
    MEMORY = "memory"


class RetrievalMode(StrEnum):
    """How Phase 5.3 retrieval ranks candidate chunks."""

    SEMANTIC = "semantic"
    BM25 = "bm25"
    HYBRID = "hybrid"


class RerankerHint(StrEnum):
    """Reranker backends for Phase 5.4."""

    CROSS_ENCODER = "cross-encoder"
    BGE = "bge"
    JINA = "jina"


class LlmProviderHint(StrEnum):
    """LLM backends for planner / reasoner / report agents."""

    GEMINI = "gemini"
    NONE = "none"


class EvaluationBackend(StrEnum):
    """Scoring backends for Phase 7.1 report evaluation."""

    AUTO = "auto"
    RAGAS = "ragas"
    DEEPEVAL = "deepeval"
    HEURISTIC = "heuristic"


class EvaluationMetric(StrEnum):
    """Standard RAG quality metrics (RAGAS / DeepEval)."""

    FAITHFULNESS = "faithfulness"
    RELEVANCY = "relevancy"
    RECALL = "recall"
    PRECISION = "precision"


__all__ = [
    "ChunkStrategy",
    "ContentType",
    "EmbeddingInputType",
    "EmbeddingProviderHint",
    "EvaluationBackend",
    "EvaluationMetric",
    "LlmProviderHint",
    "QueryIntent",
    "RerankerHint",
    "RetrievalMode",
    "SearchProviderHint",
    "VectorStoreHint",
]
