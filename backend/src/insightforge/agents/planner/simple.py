"""Deterministic planner stub used until an LLM-backed planner lands."""

from __future__ import annotations

from insightforge.agents.planner.base import Planner
from insightforge.core.exceptions import ValidationFailedError


class SimplePlanner(Planner):
    """Build a fixed research outline from the user query.

    No LLM calls — safe for local runs, CI, and graph smoke tests.
    """

    def plan(self, query: str) -> list[str]:
        cleaned = query.strip()
        if not cleaned:
            raise ValidationFailedError(
                "query must not be empty",
                details={"field": "query"},
            )

        return [
            f"Research: {cleaned}",
            "Retrieve sources",
            "Evaluate evidence",
            "Write a report",
        ]
