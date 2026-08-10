"""Graph edges — decide which node runs next."""

from __future__ import annotations

from insightforge.graph.state import GraphState

# Score needed before we stop researching and write the report.
PASS_SCORE = 0.7


def after_plan(state: GraphState) -> str:
    """Go to research, or stop if planning failed."""

    if state["errors"]:
        return "__end__"
    return "research"


def after_evaluate(state: GraphState) -> str:
    """Research again, or move on to the report.

    Continues while score is low and we still have steps left.
    """

    if state["score"] >= PASS_SCORE or state["step"] >= state["max_steps"]:
        return "report"
    return "research"
