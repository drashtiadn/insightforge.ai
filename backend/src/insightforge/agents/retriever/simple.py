"""Retriever agent — fetch relevant indexed chunks for a query."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from insightforge.agents.base import BaseAgent
from insightforge.core.logging import get_logger
from insightforge.domain.models import RetrievalHit
from insightforge.infrastructure.retrieval import RetrievalService
from insightforge.shared.enums import RetrievalMode

logger = get_logger(__name__)


class SimpleRetriever(BaseAgent):
    """Thin agent over ``RetrievalService`` (no provider SDKs)."""

    name = "retriever"

    def __init__(self, retrieval: RetrievalService) -> None:
        super().__init__()
        self._retrieval = retrieval

    def run(
        self,
        query: str,
        *,
        mode: RetrievalMode | str | None = None,
        limit: int | None = None,
        filters: Mapping[str, Any] | None = None,
        score_threshold: float | None = None,
        query_vector: Sequence[float] | None = None,
    ) -> list[RetrievalHit]:
        """Retrieve ranked context for ``query``."""

        logger.info(
            "retriever run query_len=%d mode=%s limit=%s",
            len(query),
            mode.value if isinstance(mode, RetrievalMode) else mode,
            limit,
        )
        return self._retrieval.retrieve(
            query,
            mode=mode,
            limit=limit,
            filters=filters,
            score_threshold=score_threshold,
            query_vector=query_vector,
        )
