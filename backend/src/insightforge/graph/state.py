"""Shared state that flows through every LangGraph node."""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict

# Sentinel: resolve max_steps from planned task count in plan_node.
AUTO_MAX_STEPS = 0


class GraphState(TypedDict):
    """What the research graph remembers between steps.

    Lists marked with ``Annotated[..., add]`` append on each update.
    Other fields are replaced when a node returns a new value.
    """

    query: str
    plan: list[str]
    intent: str
    tasks: list[dict[str, Any]]
    notes: Annotated[list[str], add]
    sources: Annotated[list[dict[str, str]], add]
    documents: Annotated[list[dict[str, Any]], add]
    score: float
    report: str
    errors: Annotated[list[str], add]
    step: int
    max_steps: int
    max_retries: int
    phase: str
    transitions: Annotated[list[str], add]


def initial_state(
    query: str,
    *,
    max_steps: int | None = None,
    max_retries: int = 3,
) -> GraphState:
    """Starting values for a new research run.

    ``max_steps=None`` (default) means "run every planned task" — resolved
    in ``plan_node`` once the planner returns tasks. Pass an explicit positive
    integer to cap the research budget.
    """

    return {
        "query": query,
        "plan": [],
        "intent": "",
        "tasks": [],
        "notes": [],
        "sources": [],
        "documents": [],
        "score": 0.0,
        "report": "",
        "errors": [],
        "step": 0,
        "max_steps": AUTO_MAX_STEPS if max_steps is None else max_steps,
        "max_retries": max_retries,
        "phase": "init",
        "transitions": [],
    }
