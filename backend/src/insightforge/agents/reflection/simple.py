"""Deterministic reflection agent used for offline runs and CI.

Runs a small battery of checks against the reasoner output and evidence:

- Empty / too-short synthesized answer.
- Insufficient source coverage.
- Low reasoner confidence.
- Conflicts detected between evidence clusters.

Follow-up queries are emitted for the recoverable gaps (source coverage,
conflicts) so the planner or search agent can act on them. Confidence is
computed with the shared ``research.confidence`` helpers so reasoning and
reflection use the same scale.
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
from insightforge.research import detect_conflicts, reflection_confidence

logger = get_logger(__name__)

_MIN_SOURCES = 2
_MIN_ANSWER_CHARS = 80
_MIN_REASONER_CONFIDENCE = 0.5


class SimpleReflectionAgent(ReflectionAgent):
    """Heuristic reflection with conflict awareness and confidence scoring."""

    def reflect(
        self,
        reasoning: ReasoningResult,
        *,
        hits: Sequence[RetrievalHit] | None = None,
        documents: Sequence[Document] | None = None,
    ) -> ReflectionResult:
        source_count = self._count_sources(reasoning, hits, documents)
        # Prefer the reasoner's own conflicts; fall back to computing them
        # when the reasoner does not populate the field (e.g. custom impls).
        conflicts = list(reasoning.conflicts) or detect_conflicts(reasoning.clusters)

        gaps: list[str] = []
        follow_ups: list[str] = []

        if not reasoning.answer.strip():
            gaps.append("no synthesized answer produced")
        elif len(reasoning.answer) < _MIN_ANSWER_CHARS:
            gaps.append("synthesized answer is too short to be conclusive")

        if source_count < _MIN_SOURCES:
            gaps.append(f"only {source_count} source(s) gathered")
            follow_ups.append(f"broaden search: {reasoning.query}")

        if 0 < reasoning.confidence < _MIN_REASONER_CONFIDENCE:
            gaps.append(
                f"reasoner confidence low ({reasoning.confidence:.2f} < "
                f"{_MIN_REASONER_CONFIDENCE:.2f})"
            )

        if conflicts:
            gaps.append(f"{len(conflicts)} conflicting claim(s) detected across sources")
            follow_ups.append(f"resolve contradictions on: {reasoning.query}")

        is_sufficient = not gaps
        confidence = reflection_confidence(
            is_sufficient=is_sufficient,
            source_count=source_count,
            conflict_count=len(conflicts),
            reasoning=reasoning.confidence,
        )

        if is_sufficient:
            reasoning_note = "research passes minimum gap checks"
        elif conflicts:
            reasoning_note = "conflicts and gaps detected; further verification needed"
        else:
            reasoning_note = "gaps detected; suggesting follow-up work"

        logger.info(
            "reflection verdict sufficient=%s gaps=%d conflicts=%d follow_ups=%d confidence=%.2f",
            is_sufficient,
            len(gaps),
            len(conflicts),
            len(follow_ups),
            confidence,
            extra={
                "sufficient": is_sufficient,
                "gap_count": len(gaps),
                "conflict_count": len(conflicts),
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
            conflicts=conflicts,
            reasoning=reasoning_note,
            confidence=confidence,
        )

    @staticmethod
    def _count_sources(
        reasoning: ReasoningResult,
        hits: Sequence[RetrievalHit] | None,
        documents: Sequence[Document] | None,
    ) -> int:
        if reasoning.used_source_ids:
            return len(set(reasoning.used_source_ids))
        return len(hits or []) + len(documents or [])
