"""Graph nodes — each function does one step and returns state updates.

These are deterministic stubs. Real agents plug in during later phases.
"""

from __future__ import annotations

from typing import Any

from insightforge.graph.state import GraphState


def plan_node(state: GraphState) -> dict[str, Any]:
    """Turn the query into a short research plan."""

    query = state["query"].strip()
    if not query:
        return {"errors": ["query must not be empty"]}

    return {
        "plan": [
            f"Research: {query}",
            "Retrieve sources",
            "Evaluate evidence",
            "Write a report",
        ],
    }


def research_node(state: GraphState) -> dict[str, Any]:
    """Add a research note for this step."""

    step = state["step"] + 1
    return {
        "step": step,
        "notes": [f"Step {step}: looked into {state['query']}"],
    }


def retrieve_node(state: GraphState) -> dict[str, Any]:
    """Attach a placeholder source for this step."""

    step = state["step"]
    return {
        "sources": [
            {
                "title": f"Source {step} for {state['query']}",
                "url": f"https://example.com/research/{step}",
            }
        ],
    }


def evaluate_node(state: GraphState) -> dict[str, Any]:
    """Score research quality from notes and sources (0.0-1.0)."""

    # One pass ~0.6 (below PASS_SCORE); two passes reach 1.0.
    score = min(1.0, 0.3 * len(state["notes"]) + 0.3 * len(state["sources"]))
    return {"score": score}


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
    return {"report": report}
