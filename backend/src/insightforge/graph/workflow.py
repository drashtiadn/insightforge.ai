"""Build and compile the research graph."""

from __future__ import annotations

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
from insightforge.graph.state import GraphState


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
