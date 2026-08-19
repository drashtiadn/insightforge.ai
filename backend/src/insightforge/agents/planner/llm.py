"""Gemini-backed planner with heuristic fallback."""

from __future__ import annotations

from pydantic import BaseModel, Field

from insightforge.agents.planner.analysis import analyze_query
from insightforge.agents.planner.base import Planner
from insightforge.agents.planner.schemas import QueryAnalysis, ResearchPlan
from insightforge.agents.planner.simple import SimplePlanner
from insightforge.core.exceptions import ExternalServiceError, ValidationFailedError
from insightforge.core.logging import get_logger
from insightforge.domain.models import ResearchTask
from insightforge.infrastructure.llm import LlmMessage, LlmService, parse_model
from insightforge.prompts.planner import PLANNER_SYSTEM, PLANNER_USER
from insightforge.shared.enums import QueryIntent, SearchProviderHint

logger = get_logger(__name__)


class _LlmTask(BaseModel):
    id: str = "t1"
    description: str
    search_query: str
    providers: list[str] = Field(default_factory=list)
    priority: int = Field(default=1, ge=1)


class _LlmPlanPayload(BaseModel):
    intent: str = QueryIntent.EXPLORATORY.value
    tasks: list[_LlmTask] = Field(default_factory=list)


class LlmPlanner(Planner):
    """Ask Gemini for a structured plan; fall back to ``SimplePlanner``."""

    def __init__(
        self,
        llm: LlmService,
        *,
        fallback: Planner | None = None,
    ) -> None:
        super().__init__()
        self._llm = llm
        self._fallback = fallback or SimplePlanner()

    def build_plan(self, query: str) -> ResearchPlan:
        cleaned = query.strip()
        if not cleaned:
            raise ValidationFailedError(
                "query must not be empty",
                details={"field": "query"},
            )

        if not self._llm.available:
            logger.info("llm planner unavailable; using heuristic planner")
            return self._fallback.build_plan(cleaned)

        analysis = analyze_query(cleaned)
        messages: list[LlmMessage] = [
            {"role": "system", "content": PLANNER_SYSTEM},
            {
                "role": "user",
                "content": PLANNER_USER.format(
                    query=cleaned,
                    normalized=analysis.normalized_query,
                    keywords=", ".join(analysis.keywords) or "(none)",
                ),
            },
        ]
        try:
            raw = self._llm.complete(messages, json_mode=True)
            payload = parse_model(raw, _LlmPlanPayload)
            plan = self._to_plan(cleaned, analysis, payload)
        except (ExternalServiceError, ValidationFailedError) as exc:
            logger.warning("llm planner failed; using heuristic planner error=%s", exc)
            return self._fallback.build_plan(cleaned)

        if not plan.tasks:
            logger.warning("llm planner returned no tasks; using heuristic planner")
            return self._fallback.build_plan(cleaned)
        return plan

    @staticmethod
    def _to_plan(
        query: str,
        analysis: QueryAnalysis,
        payload: _LlmPlanPayload,
    ) -> ResearchPlan:
        try:
            intent = QueryIntent(payload.intent.strip().lower())
        except ValueError:
            intent = QueryIntent.EXPLORATORY

        tasks: list[ResearchTask] = []
        for index, item in enumerate(payload.tasks, start=1):
            providers: list[SearchProviderHint] = []
            for raw in item.providers:
                try:
                    providers.append(SearchProviderHint(raw.strip().lower()))
                except ValueError:
                    continue
            if not providers:
                providers = [SearchProviderHint.WEB]
            tasks.append(
                ResearchTask(
                    id=item.id.strip() or f"t{index}",
                    description=item.description.strip() or f"Research {query}",
                    search_query=item.search_query.strip() or query,
                    providers=providers,
                    priority=item.priority,
                )
            )
        return ResearchPlan(query=query, analysis=analysis, intent=intent, tasks=tasks)
