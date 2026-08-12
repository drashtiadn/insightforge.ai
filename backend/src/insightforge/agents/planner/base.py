"""Planner agent interface — turns a query into a structured research plan."""

from __future__ import annotations

from abc import abstractmethod

from insightforge.agents.base import BaseAgent
from insightforge.agents.planner.schemas import ResearchPlan
from insightforge.agents.tools import Tool


class Planner(BaseAgent):
    """Contract for planning agents.

    Implementations analyze the query, detect intent, and decompose work into
    ordered ``ResearchTask`` items. ``plan`` remains a thin list view for
    report rendering and older call sites.
    """

    name = "planner"

    def __init__(self, tools: list[Tool] | None = None) -> None:
        super().__init__(tools=tools)

    @abstractmethod
    def build_plan(self, query: str) -> ResearchPlan:
        """Return a structured research plan for ``query``."""

    def plan(self, query: str) -> list[str]:
        """Return human-readable step descriptions from ``build_plan``."""

        return self.build_plan(query).steps

    def run(self, query: str) -> ResearchPlan:
        """``BaseAgent`` entry point — delegates to ``build_plan``."""

        return self.build_plan(query)
