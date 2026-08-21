"""Deterministic reasoner used for offline runs and CI.

Pipeline:

1. Compress raw hits/documents into a small, deduplicated evidence set.
2. Aggregate them into ``EvidenceCluster`` groups per claim.
3. Detect conflicts between clusters.
4. Score confidence from cluster strength and conflict count.
5. Synthesize a short answer from the strongest cluster claims.

The clustering and answer synthesis stay heuristic (no LLM), but the shape
of ``ReasoningResult`` — clusters, conflicts, source ids, confidence — is
what an LLM-backed reasoner will populate too, so downstream reflection
and report agents will not need changes.
"""

from __future__ import annotations

from collections.abc import Sequence

from insightforge.agents.reasoner.base import Reasoner
from insightforge.core.exceptions import ValidationFailedError
from insightforge.core.logging import get_logger
from insightforge.domain.models import (
    Conflict,
    Document,
    EvidenceCluster,
    ReasoningResult,
    RetrievalHit,
)
from insightforge.research import (
    aggregate_evidence,
    compress_documents,
    compress_hits,
    detect_conflicts,
    reasoning_confidence,
)

logger = get_logger(__name__)

_MAX_ANSWER_CHARS = 1200
_MAX_CLAIMS_IN_ANSWER = 4


class SimpleReasoner(Reasoner):
    """Heuristic reasoner: compress → aggregate → conflict-check → synthesize."""

    def __init__(
        self,
        *,
        max_hits: int = 8,
        max_documents: int = 8,
    ) -> None:
        super().__init__()
        self._max_hits = max_hits
        self._max_documents = max_documents

    def reason(
        self,
        query: str,
        *,
        hits: Sequence[RetrievalHit] | None = None,
        documents: Sequence[Document] | None = None,
        feedback: str | None = None,
    ) -> ReasoningResult:
        cleaned_query = query.strip()
        if not cleaned_query:
            logger.warning("reasoner rejected empty query")
            raise ValidationFailedError(
                "query must not be empty",
                details={"field": "query"},
            )

        compressed_hits = compress_hits(list(hits or []), max_items=self._max_hits)
        compressed_docs = compress_documents(
            list(documents or []),
            max_items=self._max_documents,
        )

        logger.info(
            "reasoner run query_len=%d hits=%d->%d documents=%d->%d",
            len(cleaned_query),
            len(hits or []),
            len(compressed_hits),
            len(documents or []),
            len(compressed_docs),
        )

        clusters = aggregate_evidence(
            hits=compressed_hits,
            documents=compressed_docs,
        )
        conflicts = detect_conflicts(clusters)
        source_ids = self._collect_source_ids(clusters)
        extra_claims = 2 if feedback and feedback.strip() else 0
        confidence = reasoning_confidence(
            clusters,
            conflicts,
            source_count=len(source_ids),
        )

        answer, key_points = self._synthesize(clusters, conflicts, extra_claims=extra_claims)

        logger.info(
            "reasoner produced clusters=%d conflicts=%d sources=%d confidence=%.2f",
            len(clusters),
            len(conflicts),
            len(source_ids),
            confidence,
            extra={
                "clusters": len(clusters),
                "conflicts": len(conflicts),
                "sources": len(source_ids),
                "confidence": confidence,
            },
        )

        return ReasoningResult(
            query=cleaned_query,
            answer=answer,
            key_points=key_points,
            used_source_ids=source_ids,
            clusters=clusters,
            conflicts=conflicts,
            confidence=confidence,
            metadata={"revision_hint": feedback.strip()} if feedback and feedback.strip() else {},
        )

    @staticmethod
    def _collect_source_ids(clusters: Sequence[EvidenceCluster]) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for cluster in clusters:
            for evidence in cluster.evidence:
                source_id = evidence.source_id
                if not source_id or source_id in seen:
                    continue
                seen.add(source_id)
                ordered.append(source_id)
        return ordered

    @staticmethod
    def _synthesize(
        clusters: Sequence[EvidenceCluster],
        conflicts: Sequence[Conflict],
        *,
        extra_claims: int = 0,
    ) -> tuple[str, list[str]]:
        if not clusters:
            return "", []

        limit = _MAX_CLAIMS_IN_ANSWER + max(0, extra_claims)
        top_claims = [cluster.claim.strip() for cluster in clusters[:limit]]
        top_claims = [claim for claim in top_claims if claim]
        if not top_claims:
            return "", []

        answer = " ".join(top_claims)
        if conflicts:
            answer += f" Note: {len(conflicts)} potential contradiction(s) detected across sources."
        if len(answer) > _MAX_ANSWER_CHARS:
            answer = answer[: _MAX_ANSWER_CHARS - 1].rstrip() + "…"

        key_points = [cluster.claim for cluster in clusters if cluster.claim]
        return answer, key_points
