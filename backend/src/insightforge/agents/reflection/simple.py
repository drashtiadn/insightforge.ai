"""Deterministic reflection agent used for offline runs and CI.

Applies a small set of heuristics over the reasoning output and its evidence.
Phase 6.2 will layer conflict detection and confidence scoring on top; the
public ``ReflectionResult`` contract stays stable.
"""

from __future__ import annotations

from collections.abc import Sequence

from insightforge.agents.reflection.base import ReflectionAgent
from insightforge.core.logging import get_logger
from insightforge.domain.models import (
    Document,
    ReasoningResult,
    ReflectionResult,
    RetrievalHit,
)

logger = get_logger(__name__)

# Minimum diverse sources before we call the research "sufficient".
_MIN_SOURCES = 2
# Minimum synthesized answer length (chars) before we accept it.
_MIN_ANSWER_CHARS = 80
# Confidence floor above which we accept the reasoner's own signal.
_MIN_REASONER_CONFIDENCE = 0.5


class SimpleReflectionAgent(ReflectionAgent):
    """Heuristic reflection: gap detection driven by evidence counts."""

    def reflect(
        self,
        reasoning: ReasoningResult,
        *,
        hits: Sequence[RetrievalHit] | None = None,
        documents: Sequence[Document] | None = None,
    ) -> ReflectionResult:
        hit_count = len(hits or [])
        doc_count = len(documents or [])
        source_count = max(hit_count + doc_count, len(reasoning.used_source_ids))

        gaps: list[str] = []
        follow_ups: list[str] = []

        if not reasoning.answer.strip():
            gaps.append("no synthesized answer produced")
        elif len(reasoning.answer) < _MIN_ANSWER_CHARS:
            gaps.append("synthesized answer is too short to be conclusive")

        if source_count < _MIN_SOURCES:
            gaps.append(f"only {source_count} source(s) gathered")
            follow_ups.append(f"broaden search: {reasoning.query}")

        if reasoning.confidence < _MIN_REASONER_CONFIDENCE and reasoning.confidence > 0:
            gaps.append(
                f"reasoner confidence low ({reasoning.confidence:.2f} < "
                f"{_MIN_REASONER_CONFIDENCE:.2f})"
            )

        is_sufficient = not gaps
        # Reflection confidence tracks source coverage but stays bounded.
        confidence = min(1.0, 0.3 + 0.15 * source_count)
        if not is_sufficient:
            confidence = max(0.0, confidence - 0.2)

        reasoning_note = (
            "research passes minimum gap checks"
            if is_sufficient
            else "gaps detected; suggesting follow-up work"
        )

        logger.info(
            "reflection verdict sufficient=%s gaps=%d follow_ups=%d confidence=%.2f",
            is_sufficient,
            len(gaps),
            len(follow_ups),
            confidence,
            extra={
                "sufficient": is_sufficient,
                "gap_count": len(gaps),
                "follow_up_count": len(follow_ups),
                "confidence": confidence,
                "source_count": source_count,
            },
        )
        for gap in gaps:
            logger.debug("reflection gap: %s", gap)

        return ReflectionResult(
            is_sufficient=is_sufficient,
            gaps=gaps,
            follow_up_queries=follow_ups,
            reasoning=reasoning_note,
            confidence=confidence,
        )
