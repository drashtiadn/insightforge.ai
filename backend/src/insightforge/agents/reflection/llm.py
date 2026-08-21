"""Gemini-backed reflection agent with heuristic fallback."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from insightforge.agents.reflection.base import ReflectionAgent
from insightforge.agents.reflection.simple import SimpleReflectionAgent
from insightforge.core.exceptions import ExternalServiceError
from insightforge.core.logging import get_logger
from insightforge.domain.models import Document, ReasoningResult, ReflectionResult, RetrievalHit
from insightforge.infrastructure.llm import LlmMessage, LlmService, parse_model
from insightforge.prompts.reflection import REFLECTION_SYSTEM, REFLECTION_USER
from insightforge.research import reflection_confidence

logger = get_logger(__name__)


class _LlmReflectionPayload(BaseModel):
    is_sufficient: bool = False
    gaps: list[str] = Field(default_factory=list)
    follow_up_queries: list[str] = Field(default_factory=list)
    reasoning: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class LlmReflectionAgent(ReflectionAgent):
    """Ask Gemini whether research is sufficient; fall back to heuristics."""

    def __init__(
        self,
        llm: LlmService,
        *,
        fallback: ReflectionAgent | None = None,
    ) -> None:
        super().__init__()
        self._llm = llm
        self._fallback = fallback or SimpleReflectionAgent()

    def reflect(
        self,
        reasoning: ReasoningResult,
        *,
        hits: Sequence[RetrievalHit] | None = None,
        documents: Sequence[Document] | None = None,
    ) -> ReflectionResult:
        heuristic = self._fallback.reflect(reasoning, hits=hits, documents=documents)
        if not self._llm.available:
            logger.info("llm reflection unavailable; using heuristic reflection")
            return heuristic

        conflicts = (
            "; ".join(f"{item.claim_a} vs {item.claim_b}" for item in reasoning.conflicts)
            or "(none)"
        )
        messages: list[LlmMessage] = [
            {"role": "system", "content": REFLECTION_SYSTEM},
            {
                "role": "user",
                "content": REFLECTION_USER.format(
                    query=reasoning.query,
                    answer=reasoning.answer or "(empty)",
                    key_points="\n".join(f"- {point}" for point in reasoning.key_points)
                    or "(none)",
                    source_count=len(set(reasoning.used_source_ids)),
                    conflicts=conflicts,
                    reasoner_confidence=f"{reasoning.confidence:.2f}",
                ),
            },
        ]
        try:
            raw = self._llm.complete(messages, json_mode=True)
            payload = parse_model(raw, _LlmReflectionPayload)
        except ExternalServiceError as exc:
            logger.warning("llm reflection failed; using heuristic reflection error=%s", exc)
            return heuristic

        gaps = [item.strip() for item in payload.gaps if item.strip()]
        follow_ups = [item.strip() for item in payload.follow_up_queries if item.strip()][:2]
        is_sufficient = bool(payload.is_sufficient) and not gaps
        source_count = len(set(reasoning.used_source_ids)) or len(hits or []) + len(documents or [])
        confidence = payload.confidence or reflection_confidence(
            is_sufficient=is_sufficient,
            source_count=source_count,
            conflict_count=len(reasoning.conflicts),
            reasoning=reasoning.confidence,
        )
        logger.info(
            "llm reflection sufficient=%s gaps=%d follow_ups=%d confidence=%.2f",
            is_sufficient,
            len(gaps),
            len(follow_ups),
            confidence,
            extra={
                "sufficient": is_sufficient,
                "gap_count": len(gaps),
                "follow_up_count": len(follow_ups),
                "confidence": confidence,
            },
        )
        return ReflectionResult(
            is_sufficient=is_sufficient,
            gaps=gaps,
            follow_up_queries=follow_ups,
            conflicts=list(reasoning.conflicts) or heuristic.conflicts,
            reasoning=payload.reasoning.strip() or heuristic.reasoning,
            confidence=confidence,
        )
