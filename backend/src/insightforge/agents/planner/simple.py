"""Offline heuristic planner: analyze → detect intent → decompose tasks."""

from __future__ import annotations

from insightforge.agents.planner.analysis import analyze_query, detect_intent
from insightforge.agents.planner.base import Planner
from insightforge.agents.planner.decompose import decompose_tasks
from insightforge.agents.planner.schemas import ResearchPlan
from insightforge.core.exceptions import ValidationFailedError
from insightforge.core.logging import get_logger

logger = get_logger(__name__)


class SimplePlanner(Planner):
    """Deterministic planner used for local runs, CI, and the default graph.

    No LLM calls. Produces provider-ready tasks so Phase 3.2 search can execute
    them without re-parsing the query.
    """

    def build_plan(self, query: str) -> ResearchPlan:
        cleaned = query.strip()
        if not cleaned:
            logger.warning("planning rejected empty query")
            raise ValidationFailedError(
                "query must not be empty",
                details={"field": "query"},
            )

        logger.info("planning started query_len=%d", len(cleaned))

        analysis = analyze_query(cleaned)
        intent = detect_intent(analysis.normalized_query)
        tasks = decompose_tasks(analysis, intent)
        plan = ResearchPlan(
            query=cleaned,
            analysis=analysis,
            intent=intent,
            tasks=tasks,
        )

        logger.info(
            "plan built intent=%s task_count=%d keywords=%d",
            intent.value,
            len(tasks),
            len(analysis.keywords),
            extra={
                "intent": intent.value,
                "task_count": len(tasks),
                "keywords": analysis.keywords,
                "task_ids": [task.id for task in tasks],
            },
        )
        for task in tasks:
            logger.debug(
                "plan task id=%s priority=%d providers=%s query=%r",
                task.id,
                task.priority,
                [provider.value for provider in task.providers],
                task.search_query,
            )

        return plan
