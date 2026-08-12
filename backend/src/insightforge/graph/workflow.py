"""Build, compile, and run the research workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

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
    retrieve_node,
)
from insightforge.graph.state import GraphState, initial_state
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


def build_graph() -> StateGraph[GraphState, None, GraphState, GraphState]:
    """Wire nodes and edges into a StateGraph.

    Flow::

        START → plan → research → retrieve → evaluate ⇄ report → END
                         ↑__________________________|
    """

    graph: StateGraph[GraphState, None, GraphState, GraphState] = StateGraph(GraphState)

    graph.add_node("plan", plan_node)
    graph.add_node("research", research_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", after_plan)
    graph.add_edge("research", "retrieve")
    graph.add_edge("retrieve", "evaluate")
    graph.add_conditional_edges("evaluate", after_evaluate)
    graph.add_edge("report", END)

    return graph


def compile_graph() -> CompiledStateGraph[GraphState, None, GraphState, GraphState]:
    """Return a runnable graph."""

    return build_graph().compile()


@traceable(name="run_research", run_type="chain", tags=["insightforge", "research"])
def run_research(
    query: str,
    *,
    max_steps: int = 2,
    max_retries: int = 3,
) -> WorkflowResult:
    """Run the full research workflow and return a structured result.

    When LangSmith tracing is enabled, this run and the LangGraph node spans
    appear in the configured project for debugging.
    """

    configure_tracing()

    state = cast(
        GraphState,
        compile_graph().invoke(initial_state(query, max_steps=max_steps, max_retries=max_retries)),
    )
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
