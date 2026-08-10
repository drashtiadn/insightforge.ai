"""Planner agent interface — turns a query into ordered research steps."""

from __future__ import annotations

from abc import abstractmethod

from insightforge.agents.base import BaseAgent
from insightforge.agents.tools import Tool


class Planner(BaseAgent):
    """Contract for planning agents.

    Implementations may call an LLM later; for now they only need to return a
    clear list of steps from ``plan``.
    """

    name = "planner"

    def __init__(self, tools: list[Tool] | None = None) -> None:
        super().__init__(tools=tools)

    @abstractmethod
    def plan(self, query: str) -> list[str]:
        """Return ordered research steps for ``query``."""

    def run(self, query: str) -> list[str]:
        """``BaseAgent`` entry point — delegates to ``plan``."""

        return self.plan(query)
