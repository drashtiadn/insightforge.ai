"""Unit tests for BM25, hybrid fusion, and RetrievalService."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from insightforge.agents.retriever import SimpleRetriever
from insightforge.core.config import Settings
from insightforge.core.exceptions import ExternalServiceError, ValidationFailedError
from insightforge.domain.models import RetrievalHit, VectorRecord
from insightforge.infrastructure.embeddings import EmbeddingService, LocalEmbeddingProvider
from insightforge.infrastructure.retrieval import (
    BM25Index,
    RetrievalService,
    create_retrieval_service,
    parse_retrieval_mode,
    reciprocal_rank_fusion,
    require_query,
)
from insightforge.infrastructure.vectorstores import MemoryVectorStore, VectorStoreService
from insightforge.shared.enums import RetrievalMode


def _record(
    record_id: str,
    text: str,
    vector: list[float],
    **metadata: object,
) -> VectorRecord:
    return VectorRecord(
        id=record_id,
        vector=vector,
        text=text,
        metadata=dict(metadata),
    )


def _keyword_embed(texts: Sequence[str]) -> list[list[float]]:
    """Map RAG/lexical cues onto a 2-D space for deterministic semantic tests."""

    vectors: list[list[float]] = []
    for text in texts:
        lowered = text.lower()
        rag = 1.0 if "rag" in lowered or "retrieval" in lowered else 0.0
        cats = 1.0 if "cat" in lowered else 0.0
        vectors.append([rag, cats])
    return vectors


def _service() -> RetrievalService:
    embeddings = EmbeddingService(
        LocalEmbeddingProvider(embed_fn=_keyword_embed, model="test-embed")
    )
    return RetrievalService(
        MemoryVectorStore(dimensions=2),
        embeddings=embeddings,
        default_mode=RetrievalMode.HYBRID,
        default_limit=5,
        candidate_multiplier=4,
        rrf_k=60,
    )


def test_require_query_rejects_blank() -> None:
    with pytest.raises(ValidationFailedError):
        require_query("  ")


def test_parse_retrieval_mode() -> None:
    assert parse_retrieval_mode("HYBRID") is RetrievalMode.HYBRID
    assert parse_retrieval_mode("bm25") is RetrievalMode.BM25
    with pytest.raises(ValidationFailedError):
        parse_retrieval_mode("keyword")


def test_bm25_ranks_lexical_match_first() -> None:
    index = BM25Index()
    index.upsert(record_id="a", text="Cats sit on mats.", metadata={"topic": "pets"})
    index.upsert(
        record_id="b",
        text="Retrieval augmented generation (RAG) uses embeddings.",
        metadata={"topic": "ml"},
    )
    index.upsert(record_id="c", text="The weather is sunny today.", metadata={"topic": "news"})

    hits = index.search("RAG retrieval embeddings", limit=3)
    assert hits[0].id == "b"
    assert hits[0].bm25_rank == 1
    assert hits[0].mode is RetrievalMode.BM25
    assert hits[0].score > 0

    filtered = index.search("RAG", limit=5, filters={"topic": "pets"})
    assert filtered == []


def test_bm25_metadata_filter() -> None:
    index = BM25Index()
    index.upsert(record_id="a", text="vector search with Qdrant", metadata={"source": "web"})
    index.upsert(record_id="b", text="vector search paper", metadata={"source": "arxiv"})
    hits = index.search("vector search", limit=5, filters={"source": "arxiv"})
    assert [hit.id for hit in hits] == ["b"]


def test_rrf_boosts_docs_in_both_lists() -> None:
    semantic = [
        RetrievalHit(id="only-dense", score=0.99, text="d", mode=RetrievalMode.SEMANTIC),
        RetrievalHit(id="both", score=0.80, text="b", mode=RetrievalMode.SEMANTIC),
    ]
    lexical = [
        RetrievalHit(id="only-lex", score=4.0, text="l", mode=RetrievalMode.BM25),
        RetrievalHit(id="both", score=3.0, text="b", mode=RetrievalMode.BM25),
    ]
    fused = reciprocal_rank_fusion(semantic, lexical, k=60, limit=3)
    assert fused[0].id == "both"
    assert fused[0].mode is RetrievalMode.HYBRID
    assert fused[0].semantic_rank == 2
    assert fused[0].bm25_rank == 2
    assert {hit.id for hit in fused} == {"both", "only-dense", "only-lex"}


def test_semantic_retrieve() -> None:
    service = _service()
    service.index(
        [
            _record("pets", "Cats purr loudly.", [0.0, 1.0], source="web"),
            _record("rag", "RAG retrieves context for an LLM.", [1.0, 0.0], source="arxiv"),
        ]
    )
    hits = service.retrieve("what is RAG retrieval?", mode=RetrievalMode.SEMANTIC, limit=2)
    assert hits[0].id == "rag"
    assert hits[0].mode is RetrievalMode.SEMANTIC
    assert hits[0].semantic_rank == 1


def test_bm25_retrieve_does_not_need_embeddings() -> None:
    service = RetrievalService(MemoryVectorStore(dimensions=2), embeddings=None)
    service.index(
        [
            _record("a", "Pineapple pizza is controversial.", [0.1, 0.1]),
            _record("b", "Hybrid search combines BM25 and dense vectors.", [0.2, 0.2]),
        ]
    )
    hits = service.retrieve("BM25 hybrid search", mode="bm25", limit=2)
    assert hits[0].id == "b"


def test_hybrid_retrieve_and_filters() -> None:
    service = _service()
    service.index(
        [
            _record("a", "Cats and RAG tutorials", [0.5, 0.5], source="web"),
            _record("b", "RAG retrieval pipelines", [1.0, 0.0], source="arxiv"),
            _record("c", "Unrelated cooking recipe", [0.0, 0.0], source="web"),
        ]
    )
    hits = service.retrieve("RAG retrieval", mode=RetrievalMode.HYBRID, limit=2)
    assert hits[0].id == "b"
    assert hits[0].mode is RetrievalMode.HYBRID

    filtered = service.retrieve(
        "RAG",
        mode=RetrievalMode.HYBRID,
        limit=5,
        filters={"source": "web"},
    )
    assert all(hit.metadata["source"] == "web" for hit in filtered)
    assert "b" not in {hit.id for hit in filtered}


def test_index_texts_and_simple_retriever() -> None:
    service = _service()
    written = service.index_texts(
        ["RAG systems retrieve documents", "Cats sleep all day"],
        ids=["d1", "d2"],
        metadata=[{"kind": "ml"}, {"kind": "pets"}],
    )
    assert written == 2
    retriever = SimpleRetriever(service)
    hits = retriever.run("retrieve documents with RAG", mode="semantic", limit=1)
    assert hits[0].id == "d1"


def test_semantic_requires_embeddings() -> None:
    service = RetrievalService(MemoryVectorStore(dimensions=2))
    service.index([_record("a", "hello world", [1.0, 0.0])])
    with pytest.raises(ExternalServiceError, match="embedding"):
        service.retrieve("hello", mode=RetrievalMode.SEMANTIC)


def test_create_retrieval_service_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECTOR_STORE", "memory")
    monkeypatch.setenv("VECTOR_DIMENSIONS", "2")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("RETRIEVAL_MODE", "hybrid")
    settings = Settings(_env_file=None)

    service = create_retrieval_service(settings, local_embed_fn=_keyword_embed)
    assert service.default_mode is RetrievalMode.HYBRID
    service.index_texts(["RAG embeddings"], ids=["x"])
    hits = service.retrieve("RAG", limit=1)
    assert hits[0].id == "x"


def test_delete_keeps_bm25_in_sync() -> None:
    service = _service()
    service.index(
        [
            _record("keep", "Keep this RAG chunk", [1.0, 0.0]),
            _record("drop", "Drop this RAG chunk", [1.0, 0.0]),
        ]
    )
    service.delete(ids=["drop"])
    hits = service.retrieve("RAG", mode="bm25", limit=5)
    assert [hit.id for hit in hits] == ["keep"]


def test_delete_ids_and_filters_keeps_bm25_aligned_with_store() -> None:
    """Vector store deletes id∩filter; BM25 must not drop ids outside the filter."""

    service = _service()
    service.index(
        [
            _record("a", "Apple fruit sweet RAG", [1.0, 0.0], doc="1"),
            _record("b", "Banana fruit yellow RAG", [0.0, 1.0], doc="2"),
            _record("c", "Carrot vegetable orange RAG", [1.0, 1.0], doc="1"),
        ]
    )

    removed = service.delete(ids=["a", "b", "c"], filters={"doc": "1"})
    assert removed == 2
    assert service.store.count() == 1

    bm25_hits = service.retrieve("Banana", mode="bm25", limit=5)
    assert [hit.id for hit in bm25_hits] == ["b"]

    # Hybrid must still surface the surviving vector via lexical (and dense).
    hybrid_ids = {hit.id for hit in service.retrieve("Banana fruit", mode="hybrid", limit=5)}
    assert "b" in hybrid_ids


def test_session_retrieval_is_isolated() -> None:
    embeddings = EmbeddingService(
        LocalEmbeddingProvider(embed_fn=_keyword_embed, model="test-embed")
    )
    store = MemoryVectorStore(dimensions=2)
    sessions = VectorStoreService(store)
    primary = RetrievalService(
        store,
        embeddings=embeddings,
        sessions=sessions,
        default_mode=RetrievalMode.BM25,
    )
    primary.index([_record("global", "Global RAG document", [1.0, 0.0])])

    session = primary.session("research-1")
    session.index([_record("local", "Session-only RAG notes", [1.0, 0.0])])

    assert [hit.id for hit in primary.retrieve("RAG", mode="bm25")] == ["global"]
    assert [hit.id for hit in session.retrieve("RAG", mode="bm25")] == ["local"]
