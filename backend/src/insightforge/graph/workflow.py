"""Build, compile, and run the research workflow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langsmith import traceable

from insightforge.core.logging import get_logger
from insightforge.graph.edges import after_evaluate, after_plan
from insightforge.graph.nodes import (
    evaluate_node,
    plan_node,
    report_node,
    research_node,
    search_node,
)
from insightforge.graph.state import GraphState, initial_state
from insightforge.infrastructure.search import SearchService, create_search_service
from insightforge.infrastructure.tracing import (
    configure_tracing,
    debug_run_summary,
    tracing_status,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class WorkflowResult:
    """Simple view of a finished research run."""

    query: str
    report: str
    score: float
    phase: str
    errors: tuple[str, ...]
    transitions: tuple[str, ...]
    state: GraphState

    @property
    def ok(self) -> bool:
        """True when the run finished a report without recorded errors."""

        return self.phase == "done" and not self.errors


def _bound_search_node(
    search_service: SearchService | None,
) -> Callable[[GraphState], dict[str, Any]]:
    """Bind ``search_service`` into ``search_node`` for LangGraph registration."""

    def _search(state: GraphState) -> dict[str, Any]:
        return search_node(state, search_service=search_service)

    return _search


def resolve_search_service(
    search_service: SearchService | None = None,
    *,
    stub_search: bool = False,
) -> tuple[SearchService | None, bool]:
    """Resolve the search service used by the default graph.

    Returns ``(service, owns_service)``. When ``stub_search`` is True, the
    offline stub path is used (``service`` is ``None``). Otherwise an omitted
    service is created via ``create_search_service()`` and marked owned.
    """

    if stub_search:
        return None, False
    if search_service is not None:
        return search_service, False
    return create_search_service(), True


def build_graph(
    *,
    search_service: SearchService | None = None,
) -> StateGraph[GraphState, None, GraphState, GraphState]:
    """Wire nodes and edges into a StateGraph.

    ``search_service`` is bound into the ``search`` node. Pass ``None`` for the
    offline stub fetcher. Prefer ``compile_graph`` / ``run_research``, which
    inject a real ``SearchService`` by default.

    Flow::

        START → plan → research → search → evaluate ⇄ report → END
                         ↑_________________________|
    """

    graph: StateGraph[GraphState, None, GraphState, GraphState] = StateGraph(GraphState)

    graph.add_node("plan", plan_node)
    graph.add_node("research", research_node)
    graph.add_node("search", cast(Any, _bound_search_node(search_service)))
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", after_plan)
    graph.add_edge("research", "search")
    graph.add_edge("search", "evaluate")
    graph.add_conditional_edges("evaluate", after_evaluate)
    graph.add_edge("report", END)

    return graph


def compile_graph(
    *,
    search_service: SearchService | None = None,
    stub_search: bool = False,
) -> CompiledStateGraph[GraphState, None, GraphState, GraphState]:
    """Return a runnable graph with ``SearchService`` injected by default.

    Prefer ``run_research`` for one-shot runs (it closes owned clients).
    For long-lived compiled graphs, pass your own ``search_service`` and call
    ``close()`` when finished. Pass ``stub_search=True`` for the offline stub.
    """

    service, _owns = resolve_search_service(search_service, stub_search=stub_search)
    return build_graph(search_service=service).compile()


@traceable(name="run_research", run_type="chain", tags=["insightforge", "research"])
def run_research(
    query: str,
    *,
    max_steps: int | None = None,
    max_retries: int = 3,
    search_service: SearchService | None = None,
    stub_search: bool = False,
) -> WorkflowResult:
    """Run the full research workflow and return a structured result.

    When LangSmith tracing is enabled, this run and the LangGraph node spans
    appear in the configured project for debugging.

    ``max_steps=None`` runs every planned research task (resolved after plan).
    A ``SearchService`` is injected by default; pass ``stub_search=True`` to
    force the offline stub fetcher.
    """

    configure_tracing()

    service, owns_service = resolve_search_service(
        search_service,
        stub_search=stub_search,
    )
    try:
        # Pass the resolved service through; stub_search avoids a second create.
        state = cast(
            GraphState,
            build_graph(search_service=service)
            .compile()
            .invoke(initial_state(query, max_steps=max_steps, max_retries=max_retries)),
        )
    finally:
        if owns_service and service is not None:
            service.close()

    result = WorkflowResult(
        query=state["query"],
        report=state["report"],
        score=state["score"],
        phase=state["phase"],
        errors=tuple(state["errors"]),
        transitions=tuple(state["transitions"]),
        state=state,
    )

    summary = debug_run_summary(
        phase=result.phase,
        ok=result.ok,
        score=result.score,
        errors=result.errors,
        transitions=result.transitions,
    )
    if tracing_status().enabled:
        logger.info(
            "research run traced query=%r phase=%s ok=%s score=%s",
            result.query,
            result.phase,
            result.ok,
            result.score,
            extra=summary,
        )
    else:
        logger.debug(
            "research run finished query=%r phase=%s ok=%s",
            result.query,
            result.phase,
            result.ok,
            extra=summary,
        )

    return result
