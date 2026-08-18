"""Deterministic reasoner used for offline runs and CI.

No LLM calls. Synthesizes a short answer by concatenating the strongest
snippets from retrieval hits and documents. Phase 6.2 replaces the naive
synthesis with evidence aggregation, conflict detection, and confidence
scoring — the contract stays the same.
"""

from __future__ import annotations

from collections.abc import Sequence

from insightforge.agents.reasoner.base import Reasoner
from insightforge.core.exceptions import ValidationFailedError
from insightforge.core.logging import get_logger
from insightforge.domain.models import Document, ReasoningResult, RetrievalHit

logger = get_logger(__name__)

# Cap synthesized answer size so downstream prompts stay predictable.
_MAX_ANSWER_CHARS = 1200
_MAX_KEY_POINTS = 6
_MAX_SNIPPET_CHARS = 240


def _snippet(text: str, *, limit: int = _MAX_SNIPPET_CHARS) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


class SimpleReasoner(Reasoner):
    """Heuristic reasoner that stitches ranked evidence into a short answer."""

    def reason(
        self,
        query: str,
        *,
        hits: Sequence[RetrievalHit] | None = None,
        documents: Sequence[Document] | None = None,
    ) -> ReasoningResult:
        cleaned_query = query.strip()
        if not cleaned_query:
            logger.warning("reasoner rejected empty query")
            raise ValidationFailedError(
                "query must not be empty",
                details={"field": "query"},
            )

        hit_list = list(hits or [])
        doc_list = list(documents or [])
        logger.info(
            "reasoner run query_len=%d hits=%d documents=%d",
            len(cleaned_query),
            len(hit_list),
            len(doc_list),
        )

        key_points: list[str] = []
        used_ids: list[str] = []

        for hit in hit_list[:_MAX_KEY_POINTS]:
            snippet = _snippet(hit.text)
            if not snippet:
                continue
            key_points.append(snippet)
            used_ids.append(hit.id)

        for doc in doc_list:
            if len(key_points) >= _MAX_KEY_POINTS:
                break
            body = doc.snippet or doc.content
            snippet = _snippet(body)
            if not snippet:
                continue
            key_points.append(snippet)
            used_ids.append(doc.url)

        if not key_points:
            logger.info("reasoner produced empty answer no_evidence=1")
            return ReasoningResult(
                query=cleaned_query,
                answer="",
                key_points=[],
                used_source_ids=[],
                confidence=0.0,
            )

        answer = " ".join(key_points)
        if len(answer) > _MAX_ANSWER_CHARS:
            answer = answer[: _MAX_ANSWER_CHARS - 1].rstrip() + "…"

        # Heuristic confidence: more sources → higher, capped at 0.9 for the
        # deterministic path. The LLM-backed reasoner will replace this.
        source_count = len(used_ids)
        confidence = min(0.9, 0.2 + 0.1 * source_count)

        logger.info(
            "reasoner produced answer key_points=%d sources=%d confidence=%.2f",
            len(key_points),
            source_count,
            confidence,
            extra={
                "key_points": len(key_points),
                "sources": source_count,
                "confidence": confidence,
            },
        )

        return ReasoningResult(
            query=cleaned_query,
            answer=answer,
            key_points=key_points,
            used_source_ids=used_ids,
            confidence=confidence,
        )
