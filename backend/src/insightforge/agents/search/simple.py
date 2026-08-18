"""Search agent — execute research tasks against the search infrastructure."""

from __future__ import annotations

from collections.abc import Sequence

from insightforge.agents.base import BaseAgent
from insightforge.core.logging import get_logger
from insightforge.domain.models import Document, ResearchTask
from insightforge.infrastructure.search import SearchService

logger = get_logger(__name__)


class SimpleSearchAgent(BaseAgent):
    """Thin agent over ``SearchService`` (no provider SDKs).

    Accepts a single ``ResearchTask``, a sequence of tasks, or a raw query
    string. Delegates execution, rate limiting, dedupe, and scoring to the
    injected ``SearchService`` so the agent stays free of transport concerns.
    """

    name = "search"

    def __init__(self, search: SearchService) -> None:
        super().__init__()
        self._search = search

    def run(
        self,
        tasks: ResearchTask | Sequence[ResearchTask],
        *,
        limit: int | None = None,
    ) -> list[Document]:
        """Run one or more ``ResearchTask`` items and return documents."""

        if isinstance(tasks, ResearchTask):
            logger.info(
                "search agent run task_id=%s providers=%s limit=%s",
                tasks.id,
                [hint.value for hint in tasks.providers],
                limit,
            )
            return self._search.search_task(tasks, limit=limit)

        task_list = list(tasks)
        logger.info(
            "search agent run task_count=%d limit=%s",
            len(task_list),
            limit,
        )
        return self._search.search_tasks(task_list, limit=limit)
