"""Report generator interface — assemble the final research artifact."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence

from insightforge.agents.base import BaseAgent
from insightforge.agents.planner.schemas import ResearchPlan
from insightforge.agents.tools import Tool
from insightforge.domain.models import (
    Document,
    ReasoningResult,
    ReflectionResult,
    ResearchReport,
    RetrievalHit,
)


class ReportGenerator(BaseAgent):
    """Contract for report generators.

    Combines the plan, reasoning output, and evidence into a
    ``ResearchReport``. Reflection input is optional — when provided its
    verdict and confidence flow through into the report metadata.
    """

    name = "report"

    def __init__(self, tools: list[Tool] | None = None) -> None:
        super().__init__(tools=tools)

    @abstractmethod
    def generate(
        self,
        *,
        plan: ResearchPlan,
        reasoning: ReasoningResult,
        reflection: ReflectionResult | None = None,
        hits: Sequence[RetrievalHit] | None = None,
        documents: Sequence[Document] | None = None,
    ) -> ResearchReport:
        """Build the final ``ResearchReport`` for a research run."""

    def run(
        self,
        *,
        plan: ResearchPlan,
        reasoning: ReasoningResult,
        reflection: ReflectionResult | None = None,
        hits: Sequence[RetrievalHit] | None = None,
        documents: Sequence[Document] | None = None,
    ) -> ResearchReport:
        """``BaseAgent`` entry point — delegates to ``generate``."""

        return self.generate(
            plan=plan,
            reasoning=reasoning,
            reflection=reflection,
            hits=hits,
            documents=documents,
        )
