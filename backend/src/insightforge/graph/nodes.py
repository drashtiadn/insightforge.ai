"""Graph nodes — each function does one step and returns state updates.

``plan_node`` uses the ``Planner`` interface. ``retrieve_node`` retries
transient failures. Unexpected errors are recorded in state so the graph can
recover instead of crashing.
"""

from __future__ import annotations

from typing import Any

from insightforge.agents import Planner, SimplePlanner
from insightforge.core.exceptions import ValidationFailedError
from insightforge.graph.retry import call_with_retry
from insightforge.graph.state import GraphState


def _move(state: GraphState, phase: str) -> dict[str, Any]:
    """Record the next phase and the transition that got us there."""

    return {
        "phase": phase,
        "transitions": [f"{state['phase']}->{phase}"],
    }


def plan_node(
    state: GraphState,
    *,
    planner: Planner | None = None,
) -> dict[str, Any]:
    """Turn the query into a short research plan via a ``Planner`` agent."""

    agent = planner or SimplePlanner()
    try:
        plan = agent.plan(state["query"])
    except ValidationFailedError as exc:
        return {"errors": [exc.message], **_move(state, "failed")}
    return {"plan": plan, **_move(state, "plan")}


def research_node(state: GraphState) -> dict[str, Any]:
    """Add a research note for this step."""

    step = state["step"] + 1
    return {
        "step": step,
        "notes": [f"Step {step}: looked into {state['query']}"],
        **_move(state, "research"),
    }


def fetch_source(query: str, step: int) -> dict[str, str]:
    """Fetch one source document.

    Isolated so tests can simulate transient failures with monkeypatch.
    """

    return {
        "title": f"Source {step} for {query}",
        "url": f"https://example.com/research/{step}",
    }


def retrieve_node(state: GraphState) -> dict[str, Any]:
    """Attach a source for this step, retrying transient I/O errors."""

    try:
        source = call_with_retry(
            lambda: fetch_source(state["query"], state["step"]),
            max_attempts=state["max_retries"],
        )
    except Exception as exc:
        # Soft failure: keep going so the graph can recover with partial data.
        return {
            "errors": [f"retrieve failed: {exc}"],
            **_move(state, "failed"),
        }

    return {"sources": [source], **_move(state, "retrieve")}


def evaluate_node(state: GraphState) -> dict[str, Any]:
    """Score research quality from notes and sources (0.0-1.0)."""

    # One pass ~0.6 (below PASS_SCORE); two passes reach 1.0.
    score = min(1.0, 0.3 * len(state["notes"]) + 0.3 * len(state["sources"]))
    return {"score": score, **_move(state, "evaluate")}


def report_node(state: GraphState) -> dict[str, Any]:
    """Build a short report from plan, notes, and sources."""

    plan = "\n".join(f"- {item}" for item in state["plan"]) or "- none"
    notes = "\n".join(f"- {note}" for note in state["notes"]) or "- none"
    sources = "\n".join(f"- [{s['title']}]({s['url']})" for s in state["sources"]) or "- none"

    report = (
        f"Query: {state['query']}\n"
        f"Score: {state['score']}\n\n"
        f"Plan:\n{plan}\n\n"
        f"Notes:\n{notes}\n\n"
        f"Sources:\n{sources}\n"
    )
    if state["errors"]:
        error_lines = "\n".join(f"- {err}" for err in state["errors"])
        report += f"\nErrors:\n{error_lines}\n"

    return {"report": report, **_move(state, "done")}
