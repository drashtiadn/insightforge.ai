"""Hybrid fusion — Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from collections.abc import Sequence

from insightforge.core.logging import get_logger
from insightforge.domain.models import RetrievalHit
from insightforge.shared.enums import RetrievalMode

logger = get_logger(__name__)

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    semantic: Sequence[RetrievalHit],
    lexical: Sequence[RetrievalHit],
    *,
    k: int = DEFAULT_RRF_K,
    limit: int = 5,
) -> list[RetrievalHit]:
    """Merge two ranked lists with RRF: ``1 / (k + rank)``.

    ``k`` (default 60) dampens the impact of any single list. Documents that
    appear in both lists rank above documents that appear in only one.
    """

    if limit < 1:
        return []

    scores: dict[str, float] = {}
    texts: dict[str, str] = {}
    metadata: dict[str, dict[str, object]] = {}
    semantic_rank: dict[str, int] = {}
    bm25_rank: dict[str, int] = {}

    for rank, hit in enumerate(semantic, start=1):
        scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank)
        texts.setdefault(hit.id, hit.text)
        metadata.setdefault(hit.id, dict(hit.metadata))
        semantic_rank[hit.id] = rank

    for rank, hit in enumerate(lexical, start=1):
        scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank)
        texts.setdefault(hit.id, hit.text)
        metadata.setdefault(hit.id, dict(hit.metadata))
        bm25_rank[hit.id] = rank

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    fused: list[RetrievalHit] = []
    for record_id, score in ordered[:limit]:
        fused.append(
            RetrievalHit(
                id=record_id,
                score=score,
                text=texts.get(record_id, ""),
                metadata=dict(metadata.get(record_id) or {}),
                mode=RetrievalMode.HYBRID,
                semantic_rank=semantic_rank.get(record_id),
                bm25_rank=bm25_rank.get(record_id),
            )
        )

    logger.debug(
        "rrf fused semantic=%d bm25=%d final=%d k=%d",
        len(semantic),
        len(lexical),
        len(fused),
        k,
        extra={"final": len(fused), "rrf_k": k},
    )
    return fused
