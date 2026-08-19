"""Graph nodes — each function does one step and returns state updates.

``plan_node`` uses the ``Planner`` interface. ``search_node`` can pull real
documents via ``SearchService`` when tasks exist; otherwise it uses the stub
fetcher so CI stays offline. Unexpected errors are recorded in state so the
graph can recover instead of crashing.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from insightforge.agents import Planner, SimplePlanner
from insightforge.core.exceptions import ValidationFailedError
from insightforge.core.logging import get_logger
from insightforge.domain.models import ResearchTask
from insightforge.graph.helpers import move
from insightforge.graph.retry import call_with_retry
from insightforge.graph.state import AUTO_MAX_STEPS, GraphState
from insightforge.infrastructure.search import SearchService
from insightforge.shared.enums import SearchProviderHint

logger = get_logger(__name__)


def _move(state: GraphState, phase: str) -> dict[str, Any]:
    """Record the next phase and the transition that got us there."""

    return move(state, phase)


def plan_node(
    state: GraphState,
    *,
    planner: Planner | None = None,
) -> dict[str, Any]:
    """Analyze the query and store a structured research plan."""

    agent = planner or SimplePlanner()
    try:
        research_plan = agent.build_plan(state["query"])
    except ValidationFailedError as exc:
        logger.warning(
            "plan validation failed query_len=%d error=%s",
            len(state["query"]),
            exc.message,
        )
        return {"errors": [exc.message], **_move(state, "failed")}

    task_count = len(research_plan.tasks)
    # Auto budget: run every planned task. Explicit max_steps stays a hard cap.
    if state["max_steps"] == AUTO_MAX_STEPS:
        resolved_max_steps = task_count or 1
    else:
        resolved_max_steps = state["max_steps"]

    logger.info(
        "plan node complete intent=%s task_count=%d max_steps=%d",
        research_plan.intent.value,
        task_count,
        resolved_max_steps,
        extra={
            "intent": research_plan.intent.value,
            "task_count": task_count,
            "max_steps": resolved_max_steps,
        },
    )
    return {
        "plan": research_plan.steps,
        "intent": research_plan.intent.value,
        "tasks": [task.model_dump(mode="json") for task in research_plan.tasks],
        "max_steps": resolved_max_steps,
        **_move(state, "plan"),
    }


def research_node(state: GraphState) -> dict[str, Any]:
    """Add a research note for this step."""

    step = state["step"] + 1
    note = f"Step {step}: looked into {state['query']}"
    if state["tasks"]:
        idx = min(step - 1, len(state["tasks"]) - 1)
        task_desc = str(state["tasks"][idx].get("description") or "")
        if task_desc:
            note = f"Step {step}: {task_desc}"
    return {
        "step": step,
        "notes": [note],
        **_move(state, "research"),
    }


def fetch_source(query: str, step: int) -> dict[str, str]:
    """Fetch one source document.

    Isolated so tests can simulate transient failures with monkeypatch.
    Offline stub used when no ``SearchService`` is injected.
    """

    return {
        "title": f"Source {step} for {query}",
        "url": f"https://example.com/research/{step}",
        "snippet": (
            f"An overview of {query} covering definitions, history, "
            f"current practice, and open questions. Source {step}."
        ),
    }


def _task_for_step(state: GraphState) -> ResearchTask | None:
    tasks = state["tasks"]
    if not tasks:
        return None
    idx = min(max(state["step"] - 1, 0), len(tasks) - 1)
    try:
        return ResearchTask.model_validate(tasks[idx])
    except ValidationError as exc:
        logger.warning("invalid research task at index=%d error=%s", idx, exc)
        return None


def search_node(
    state: GraphState,
    *,
    search_service: SearchService | None = None,
) -> dict[str, Any]:
    """Run external search (or the offline stub) for this step's task.

    When ``search_service`` is provided and planner tasks exist, runs the
    Phase 3 search pipeline (parallel / dedupe / score / rate-limit).
    Otherwise falls back to the offline stub fetcher.
    """

    task = _task_for_step(state) if search_service is not None else None
    if search_service is not None and task is not None:
        try:
            documents = call_with_retry(
                lambda: search_service.search_task(task),
                max_attempts=state["max_retries"],
            )
        except Exception as exc:
            return {
                "errors": [f"search failed: {exc}"],
                **_move(state, "failed"),
            }

        serialized = [doc.model_dump(mode="json") for doc in documents]
        sources = [{"title": doc.title, "url": doc.url} for doc in documents]
        logger.info(
            "search node task_id=%s document_count=%d",
            task.id,
            len(documents),
        )
        return {
            "documents": serialized,
            "sources": sources,
            **_move(state, "search"),
        }

    try:
        source = call_with_retry(
            lambda: fetch_source(state["query"], state["step"]),
            max_attempts=state["max_retries"],
        )
    except Exception as exc:
        # Soft failure: keep going so the graph can recover with partial data.
        return {
            "errors": [f"search failed: {exc}"],
            **_move(state, "failed"),
        }

    document = {
        "title": source["title"],
        "url": source["url"],
        "snippet": source.get("snippet") or source["title"],
        "content": source.get("snippet") or source["title"],
        "provider": SearchProviderHint.WEB.value,
    }
    return {
        "sources": [{"title": source["title"], "url": source["url"]}],
        "documents": [document],
        **_move(state, "search"),
    }
