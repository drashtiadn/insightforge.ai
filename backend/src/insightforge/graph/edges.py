"""Graph edges — decide which node runs next."""

from __future__ import annotations

from insightforge.graph.state import GraphState

# Used as a fallback gate when reflection is missing.
PASS_SCORE = 0.7


def _has_partial(state: GraphState) -> bool:
    return bool(
        state["plan"]
        or state["notes"]
        or state["sources"]
        or state["documents"]
        or state["chunks"]
        or state["reasoning"]
    )


def after_plan(state: GraphState) -> str:
    """Go to research, or stop if planning failed with nothing to recover."""

    if state["errors"] or state["phase"] == "failed":
        return "__end__"
    return "research"


def after_search(state: GraphState) -> str:
    """Gather remaining planned tasks, then ingest. Recover with partial data."""

    if state["errors"] and not _has_partial(state):
        return "__end__"

    if state["step"] >= state["max_steps"]:
        return "ingest"

    tasks = state["tasks"]
    if tasks and state["step"] < len(tasks):
        return "research"

    return "ingest"


def after_reflect(state: GraphState) -> str:
    """Report, or run a follow-up search when reflection scheduled more work."""

    if state["errors"] and not _has_partial(state):
        return "__end__"
    if state["errors"]:
        return "report"

    if (
        state["step"] < state["max_steps"]
        and state["tasks"]
        and state["step"] < len(state["tasks"])
    ):
        return "research"
    return "report"


def after_judge(state: GraphState) -> str:
    """Retry reasoning when the judge rejects the answer, otherwise finish."""

    if state["phase"] == "judge":
        return "reason"
    return "__end__"
