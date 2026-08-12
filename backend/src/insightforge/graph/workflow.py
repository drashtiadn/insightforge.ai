"""Build, compile, and run the research workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from insightforge.graph.edges import after_evaluate, after_plan
from insightforge.graph.nodes import (
    evaluate_node,
    plan_node,
    report_node,
    research_node,
    retrieve_node,
)
from insightforge.graph.state import GraphState, initial_state


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


def run_research(
    query: str,
    *,
    max_steps: int = 2,
    max_retries: int = 3,
) -> WorkflowResult:
    """Run the full research workflow and return a structured result."""

    state = cast(
        GraphState,
        compile_graph().invoke(initial_state(query, max_steps=max_steps, max_retries=max_retries)),
    )
    return WorkflowResult(
        query=state["query"],
        report=state["report"],
        score=state["score"],
        phase=state["phase"],
        errors=tuple(state["errors"]),
        transitions=tuple(state["transitions"]),
        state=state,
    )
