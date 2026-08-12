"""Shared state that flows through every LangGraph node."""

from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict


class GraphState(TypedDict):
    """What the research graph remembers between steps.

    Lists marked with ``Annotated[..., add]`` append on each update.
    Other fields are replaced when a node returns a new value.
    """

    query: str
    plan: list[str]
    notes: Annotated[list[str], add]
    sources: Annotated[list[dict[str, str]], add]
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
    max_steps: int = 2,
    max_retries: int = 3,
) -> GraphState:
    """Starting values for a new research run."""

    return {
        "query": query,
        "plan": [],
        "notes": [],
        "sources": [],
        "score": 0.0,
        "report": "",
        "errors": [],
        "step": 0,
        "max_steps": max_steps,
        "max_retries": max_retries,
        "phase": "init",
        "transitions": [],
    }
