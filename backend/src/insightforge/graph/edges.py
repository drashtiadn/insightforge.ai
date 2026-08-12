"""Graph edges — decide which node runs next."""

from __future__ import annotations

from insightforge.graph.state import GraphState

# Score needed before we stop researching and write the report (only after
# planned tasks are exhausted, or when no structured tasks exist).
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
    3. Research budget exhausted → report
    4. Planned tasks remain within budget → research again
    5. Score met with no remaining planned work → report
    6. Otherwise → research again
    """

    has_partial = bool(state["plan"] or state["notes"] or state["sources"])
    if state["errors"] and has_partial:
        return "report"
    if state["errors"]:
        return "__end__"

    if state["step"] >= state["max_steps"]:
        return "report"

    tasks = state["tasks"]
    if tasks and state["step"] < len(tasks):
        # Execute every planned task up to the budget before score early-exit.
        return "research"

    if state["score"] >= PASS_SCORE:
        return "report"
    return "research"
