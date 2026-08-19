"""Small helpers shared by graph nodes without importing the package graph."""

from __future__ import annotations

from typing import Any

from insightforge.graph.state import GraphState


def move(state: GraphState, phase: str) -> dict[str, Any]:
    """Record the next phase and the transition that got us there."""

    return {
        "phase": phase,
        "transitions": [f"{state['phase']}->{phase}"],
    }
