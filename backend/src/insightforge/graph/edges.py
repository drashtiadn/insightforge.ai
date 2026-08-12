"""Graph edges — decide which node runs next."""

from __future__ import annotations

from insightforge.graph.state import GraphState

# Score needed before we stop researching and write the report.
PASS_SCORE = 0.7


def after_plan(state: GraphState) -> str:
    """Go to research, or stop if planning failed with nothing to recover."""

    if state["errors"] or state["phase"] == "failed":
        return "__end__"
    return "research"


def after_evaluate(state: GraphState) -> str:
    """Choose the next step after evaluation.

    Order of decisions:
    1. Soft failure with partial data → report (error recovery)
    2. Hard failure with no useful data → end
    3. Score/budget met → report
    4. Otherwise → research again
    """

    has_partial = bool(state["plan"] or state["notes"] or state["sources"])
    if state["errors"] and has_partial:
        return "report"
    if state["errors"]:
        return "__end__"
    if state["score"] >= PASS_SCORE or state["step"] >= state["max_steps"]:
        return "report"
    return "research"
