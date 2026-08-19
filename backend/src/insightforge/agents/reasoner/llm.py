"""Gemini-backed reasoner with heuristic fallback."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from insightforge.agents.reasoner.base import Reasoner
from insightforge.agents.reasoner.simple import SimpleReasoner
from insightforge.core.exceptions import ExternalServiceError, ValidationFailedError
from insightforge.core.logging import get_logger
from insightforge.domain.models import Document, ReasoningResult, RetrievalHit
from insightforge.infrastructure.llm import LlmMessage, LlmService, parse_model
from insightforge.prompts.reasoner import REASONER_SYSTEM, REASONER_USER
from insightforge.research import (
    aggregate_evidence,
    compress_documents,
    compress_hits,
    detect_conflicts,
    reasoning_confidence,
)

logger = get_logger(__name__)


class _LlmReasoningPayload(BaseModel):
    answer: str = ""
    key_points: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class LlmReasoner(Reasoner):
    """Cluster evidence locally, then ask Gemini to synthesize the answer."""

    def __init__(
        self,
        llm: LlmService,
        *,
        fallback: Reasoner | None = None,
        max_hits: int = 8,
        max_documents: int = 8,
    ) -> None:
        super().__init__()
        self._llm = llm
        self._fallback = fallback or SimpleReasoner(
            max_hits=max_hits,
            max_documents=max_documents,
        )
        self._max_hits = max_hits
        self._max_documents = max_documents

    def reason(
        self,
        query: str,
        *,
        hits: Sequence[RetrievalHit] | None = None,
        documents: Sequence[Document] | None = None,
    ) -> ReasoningResult:
        cleaned = query.strip()
        if not cleaned:
            raise ValidationFailedError(
                "query must not be empty",
                details={"field": "query"},
            )

        if not self._llm.available:
            logger.info("llm reasoner unavailable; using heuristic reasoner")
            return self._fallback.reason(cleaned, hits=hits, documents=documents)

        heuristic = SimpleReasoner(
            max_hits=self._max_hits,
            max_documents=self._max_documents,
        )
        base = heuristic.reason(cleaned, hits=hits, documents=documents)
        compressed_hits = compress_hits(list(hits or []), max_items=self._max_hits)
        compressed_docs = compress_documents(
            list(documents or []),
            max_items=self._max_documents,
        )
        clusters = list(base.clusters) or aggregate_evidence(
            hits=compressed_hits,
            documents=compressed_docs,
        )
        conflicts = list(base.conflicts) or detect_conflicts(clusters)

        cluster_lines = []
        for index, cluster in enumerate(clusters, start=1):
            cluster_lines.append(f"{index}. {cluster.claim}")
        conflict_lines = [
            f"- {conflict.reason}: {conflict.claim_a} vs {conflict.claim_b}"
            for conflict in conflicts
        ]
        messages: list[LlmMessage] = [
            {"role": "system", "content": REASONER_SYSTEM},
            {
                "role": "user",
                "content": REASONER_USER.format(
                    query=cleaned,
                    clusters="\n".join(cluster_lines) or "(none)",
                    conflicts="\n".join(conflict_lines) or "(none)",
                ),
            },
        ]
        try:
            raw = self._llm.complete(messages, json_mode=True)
            payload = parse_model(raw, _LlmReasoningPayload)
        except ExternalServiceError as exc:
            logger.warning("llm reasoner failed; using heuristic reasoner error=%s", exc)
            return base

        answer = payload.answer.strip() or base.answer
        key_points = [point.strip() for point in payload.key_points if point.strip()] or list(
            base.key_points
        )
        confidence = payload.confidence or reasoning_confidence(
            clusters,
            conflicts,
            source_count=len(base.used_source_ids),
        )
        return base.model_copy(
            update={
                "answer": answer,
                "key_points": key_points,
                "confidence": confidence,
            }
        )
