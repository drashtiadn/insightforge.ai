"""Retrieval service — semantic, BM25, and hybrid search over indexed chunks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from insightforge.core.config import Settings, get_settings
from insightforge.core.exceptions import ExternalServiceError, ValidationFailedError
from insightforge.core.logging import get_logger
from insightforge.domain.models import RetrievalHit, VectorHit, VectorRecord
from insightforge.infrastructure.embeddings.providers import LocalEmbedFn
from insightforge.infrastructure.embeddings.service import (
    EmbeddingService,
    create_embedding_service,
)
from insightforge.infrastructure.retrieval.bm25 import BM25Index
from insightforge.infrastructure.retrieval.fusion import reciprocal_rank_fusion
from insightforge.infrastructure.vectorstores.base import VectorStore
from insightforge.infrastructure.vectorstores.service import (
    VectorStoreService,
    create_vector_store_service,
)
from insightforge.shared.enums import RetrievalMode

logger = get_logger(__name__)


def parse_retrieval_mode(value: str) -> RetrievalMode:
    """Parse settings/env value into ``RetrievalMode``."""

    cleaned = value.strip().lower()
    try:
        return RetrievalMode(cleaned)
    except ValueError as exc:
        allowed = ", ".join(mode.value for mode in RetrievalMode)
        raise ValidationFailedError(
            f"unknown retrieval mode {value!r}; expected one of: {allowed}",
            details={"field": "retrieval_mode", "value": value},
        ) from exc


def require_query(query: str) -> str:
    """Normalize and reject blank retrieval queries."""

    cleaned = query.strip()
    if not cleaned:
        raise ValidationFailedError(
            "retrieval query must not be empty",
            details={"field": "query"},
        )
    return cleaned


def _from_vector_hit(hit: VectorHit, *, rank: int) -> RetrievalHit:
    return RetrievalHit(
        id=hit.id,
        score=hit.score,
        text=hit.text,
        metadata=dict(hit.metadata),
        mode=RetrievalMode.SEMANTIC,
        semantic_rank=rank,
    )


class RetrievalService:
    """Index chunks once, retrieve relevant context for any query.

    - **semantic** — embed the query, dense kNN in the vector store
    - **bm25** — Okapi BM25 over indexed text
    - **hybrid** — both lists fused with Reciprocal Rank Fusion

    Metadata equality filters apply to every mode. The BM25 corpus is a
    process-local sidecar: always index through this service so lexical and
    dense indexes stay aligned.
    """

    def __init__(
        self,
        store: VectorStore,
        *,
        embeddings: EmbeddingService | None = None,
        bm25: BM25Index | None = None,
        default_mode: RetrievalMode = RetrievalMode.HYBRID,
        default_limit: int = 8,
        candidate_multiplier: int = 4,
        rrf_k: int = 60,
        sessions: VectorStoreService | None = None,
    ) -> None:
        self._store = store
        self._embeddings = embeddings
        self._bm25 = bm25 or BM25Index()
        self._default_mode = default_mode
        self._default_limit = max(1, default_limit)
        self._candidate_multiplier = max(1, candidate_multiplier)
        self._rrf_k = max(1, rrf_k)
        self._sessions = sessions

    @property
    def store(self) -> VectorStore:
        return self._store

    @property
    def embeddings(self) -> EmbeddingService | None:
        return self._embeddings

    @property
    def default_mode(self) -> RetrievalMode:
        return self._default_mode

    def index(self, records: Sequence[VectorRecord]) -> int:
        """Upsert records into the vector store and the BM25 corpus."""

        written = self._store.upsert(records)
        for record in records:
            self._bm25.upsert(
                record_id=record.id,
                text=record.text,
                metadata=record.metadata,
            )
        logger.info(
            "retrieval index upserted=%d bm25_corpus=%d",
            written,
            len(self._bm25),
            extra={"upserted": written, "bm25_corpus": len(self._bm25)},
        )
        return written

    def index_texts(
        self,
        texts: Sequence[str],
        *,
        ids: Sequence[str] | None = None,
        metadata: Sequence[Mapping[str, Any]] | None = None,
    ) -> int:
        """Embed ``texts`` then index them. Requires an embedding provider."""

        cleaned = [text.strip() for text in texts]
        if any(not text for text in cleaned):
            raise ValidationFailedError(
                "indexed texts must not be empty",
                details={"field": "texts"},
            )
        vectors = self._embed_documents(cleaned)
        record_ids = list(ids) if ids is not None else [str(index) for index in range(len(cleaned))]
        if len(record_ids) != len(cleaned):
            raise ValidationFailedError(
                "ids length must match texts",
                details={"ids": len(record_ids), "texts": len(cleaned)},
            )
        meta_list: list[Mapping[str, Any]]
        if metadata is None:
            meta_list = [{} for _ in cleaned]
        else:
            meta_list = list(metadata)
            if len(meta_list) != len(cleaned):
                raise ValidationFailedError(
                    "metadata length must match texts",
                    details={"metadata": len(meta_list), "texts": len(cleaned)},
                )
        records = [
            VectorRecord(
                id=record_id,
                vector=list(vector),
                text=text,
                metadata=dict(meta),
            )
            for record_id, vector, text, meta in zip(
                record_ids, vectors, cleaned, meta_list, strict=True
            )
        ]
        return self.index(records)

    def delete(
        self,
        *,
        ids: Sequence[str] | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> int:
        """Delete from both the vector store and BM25."""

        removed = self._store.delete(ids=ids, filters=filters)
        if ids is not None:
            self._bm25.delete(ids)
        elif filters is not None:
            self._bm25.delete_matching(filters)
        logger.info(
            "retrieval delete removed=%d bm25_corpus=%d",
            removed,
            len(self._bm25),
            extra={"removed": removed},
        )
        return removed

    def clear(self) -> None:
        self._store.clear()
        self._bm25.clear()
        logger.info("retrieval indexes cleared")

    def retrieve(
        self,
        query: str,
        *,
        mode: RetrievalMode | str | None = None,
        limit: int | None = None,
        filters: Mapping[str, Any] | None = None,
        score_threshold: float | None = None,
        query_vector: Sequence[float] | None = None,
    ) -> list[RetrievalHit]:
        """Return the most relevant indexed chunks for ``query``."""

        cleaned = require_query(query)
        resolved = (
            parse_retrieval_mode(mode) if isinstance(mode, str) else (mode or self._default_mode)
        )
        top_k = limit if limit is not None else self._default_limit
        if top_k < 1:
            return []

        logger.info(
            "retrieve started mode=%s query_len=%d limit=%d filters=%s",
            resolved.value,
            len(cleaned),
            top_k,
            sorted(filters.keys()) if filters else [],
            extra={"mode": resolved.value, "limit": top_k, "query_len": len(cleaned)},
        )

        if resolved is RetrievalMode.SEMANTIC:
            hits = self._semantic(
                cleaned,
                limit=top_k,
                filters=filters,
                score_threshold=score_threshold,
                query_vector=query_vector,
            )
        elif resolved is RetrievalMode.BM25:
            hits = self._bm25.search(
                cleaned,
                limit=top_k,
                filters=filters,
                score_threshold=score_threshold,
            )
        else:
            hits = self._hybrid(
                cleaned,
                limit=top_k,
                filters=filters,
                score_threshold=score_threshold,
                query_vector=query_vector,
            )

        logger.info(
            "retrieve finished mode=%s hits=%d",
            resolved.value,
            len(hits),
            extra={"mode": resolved.value, "hits": len(hits)},
        )
        return hits

    def session(self, session_id: str) -> RetrievalService:
        """Isolated retrieval over a TTL session vector store."""

        if self._sessions is None:
            raise ValidationFailedError(
                "session retrieval requires a VectorStoreService",
                details={"field": "session_id"},
            )
        store = self._sessions.session(session_id)
        logger.info(
            "retrieval session store session_id=%s",
            session_id,
            extra={"session_id": session_id},
        )
        return RetrievalService(
            store,
            embeddings=self._embeddings,
            bm25=BM25Index(k1=self._bm25.k1, b=self._bm25.b),
            default_mode=self._default_mode,
            default_limit=self._default_limit,
            candidate_multiplier=self._candidate_multiplier,
            rrf_k=self._rrf_k,
        )

    def _hybrid(
        self,
        query: str,
        *,
        limit: int,
        filters: Mapping[str, Any] | None,
        score_threshold: float | None,
        query_vector: Sequence[float] | None,
    ) -> list[RetrievalHit]:
        candidates = max(limit, limit * self._candidate_multiplier)
        semantic = self._semantic(
            query,
            limit=candidates,
            filters=filters,
            score_threshold=score_threshold,
            query_vector=query_vector,
        )
        lexical = self._bm25.search(query, limit=candidates, filters=filters)
        return reciprocal_rank_fusion(semantic, lexical, k=self._rrf_k, limit=limit)

    def _semantic(
        self,
        query: str,
        *,
        limit: int,
        filters: Mapping[str, Any] | None,
        score_threshold: float | None,
        query_vector: Sequence[float] | None,
    ) -> list[RetrievalHit]:
        vector = list(query_vector) if query_vector is not None else self._embed_query(query)
        raw = self._store.search(
            vector,
            limit=limit,
            filters=filters,
            score_threshold=score_threshold,
        )
        return [_from_vector_hit(hit, rank=index) for index, hit in enumerate(raw, start=1)]

    def _embed_query(self, query: str) -> list[float]:
        if self._embeddings is None or not self._embeddings.available:
            raise ExternalServiceError(
                "semantic retrieval requires an available embedding provider",
                details={"mode": RetrievalMode.SEMANTIC.value},
            )
        return self._embeddings.embed_query(query)

    def _embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if self._embeddings is None or not self._embeddings.available:
            raise ExternalServiceError(
                "index_texts requires an available embedding provider",
                details={"field": "embeddings"},
            )
        result = self._embeddings.embed_documents(texts)
        if len(result.vectors) != len(texts):
            raise ExternalServiceError(
                "embedding provider returned unexpected vector count",
                details={"expected": len(texts), "actual": len(result.vectors)},
            )
        return result.vectors


def create_retrieval_service(
    settings: Settings | None = None,
    *,
    store_service: VectorStoreService | None = None,
    embeddings: EmbeddingService | None = None,
    local_embed_fn: LocalEmbedFn | None = None,
    qdrant_client: Any | None = None,
) -> RetrievalService:
    """Factory used by application code and tests."""

    cfg = settings or get_settings()
    vectors = store_service or create_vector_store_service(cfg, qdrant_client=qdrant_client)
    embedder = embeddings
    if embedder is None:
        embedder = create_embedding_service(cfg, local_embed_fn=local_embed_fn)

    mode = parse_retrieval_mode(cfg.retrieval_mode)
    logger.info(
        "retrieval service created mode=%s limit=%d store=%s embedder=%s",
        mode.value,
        cfg.retrieval_default_limit,
        vectors.store.name.value,
        embedder.provider.name.value if embedder.available else "unavailable",
        extra={
            "mode": mode.value,
            "store": vectors.store.name.value,
            "embedder_available": embedder.available,
        },
    )
    return RetrievalService(
        vectors.store,
        embeddings=embedder,
        bm25=BM25Index(k1=cfg.retrieval_bm25_k1, b=cfg.retrieval_bm25_b),
        default_mode=mode,
        default_limit=cfg.retrieval_default_limit,
        candidate_multiplier=cfg.retrieval_candidate_multiplier,
        rrf_k=cfg.retrieval_rrf_k,
        sessions=vectors,
    )
