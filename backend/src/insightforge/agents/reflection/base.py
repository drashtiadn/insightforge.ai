"""Reflection agent interface — judge whether research is good enough."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence

from insightforge.agents.base import BaseAgent
from insightforge.agents.tools import Tool
from insightforge.domain.models import (
    Document,
    ReasoningResult,
    ReflectionResult,
    RetrievalHit,
)


class ReflectionAgent(BaseAgent):
    """Contract for reflection agents.

    Given the reasoner's output and the evidence it saw, decide whether the
    workflow may proceed to report generation. When gaps exist, propose
    concrete follow-up queries so the planner or search agent can act.
    """

    name = "reflection"

    def __init__(self, tools: list[Tool] | None = None) -> None:
        super().__init__(tools=tools)

    @abstractmethod
    def reflect(
        self,
        reasoning: ReasoningResult,
        *,
        hits: Sequence[RetrievalHit] | None = None,
        documents: Sequence[Document] | None = None,
    ) -> ReflectionResult:
        """Return a ``ReflectionResult`` for the current research state."""

    def run(
        self,
        reasoning: ReasoningResult,
        *,
        hits: Sequence[RetrievalHit] | None = None,
        documents: Sequence[Document] | None = None,
    ) -> ReflectionResult:
        """``BaseAgent`` entry point — delegates to ``reflect``."""

        return self.reflect(reasoning, hits=hits, documents=documents)
