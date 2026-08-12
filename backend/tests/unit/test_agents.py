"""Tests for agent interfaces (base, planner, tools)."""

from __future__ import annotations

import pytest

from insightforge.agents import (
    BaseAgent,
    FunctionTool,
    Planner,
    ResearchPlan,
    ResearchTask,
    SimplePlanner,
    Tool,
)
from insightforge.agents.planner.schemas import QueryAnalysis
from insightforge.core.exceptions import ValidationFailedError
from insightforge.graph.nodes import plan_node
from insightforge.graph.state import initial_state
from insightforge.shared.enums import QueryIntent, SearchProviderHint


def test_simple_planner_builds_structured_plan() -> None:
    planner = SimplePlanner()
    plan = planner.build_plan("climate models")

    assert planner.name == "planner"
    assert isinstance(planner, BaseAgent)
    assert isinstance(planner, Planner)
    assert isinstance(plan, ResearchPlan)
    assert len(plan.tasks) == 3
    assert "climate" in plan.analysis.normalized_query.lower()


def test_simple_planner_run_matches_build_plan() -> None:
    planner = SimplePlanner()
    assert planner.run("AI safety") == planner.build_plan("AI safety")


def test_simple_planner_rejects_empty_query() -> None:
    with pytest.raises(ValidationFailedError) as exc_info:
        SimplePlanner().plan("   ")

    assert exc_info.value.message == "query must not be empty"
    assert exc_info.value.details == {"field": "query"}


def test_function_tool_runs_wrapped_function() -> None:
    tool: Tool = FunctionTool(
        name="echo",
        description="Return the text unchanged",
        func=lambda text: text,
    )

    assert tool.name == "echo"
    assert tool.run(text="hello") == "hello"


def test_base_agent_get_tool() -> None:
    echo = FunctionTool("echo", "echo text", lambda text: text.upper())
    planner = SimplePlanner(tools=[echo])

    assert planner.get_tool("echo").run(text="hi") == "HI"
    with pytest.raises(KeyError, match="tool not found: missing"):
        planner.get_tool("missing")


def test_plan_node_uses_planner_interface() -> None:
    result = plan_node(initial_state("multi-agent systems"))
    assert result["intent"] == QueryIntent.EXPLORATORY.value
    assert len(result["plan"]) == 3
    assert "multi-agent systems" in result["plan"][0]


def test_plan_node_accepts_custom_planner() -> None:
    class FixedPlanner(Planner):
        def build_plan(self, query: str) -> ResearchPlan:
            analysis = QueryAnalysis(
                original_query=query,
                normalized_query=query.strip(),
                keywords=[query.strip()],
                token_count=1,
            )
            tasks = [
                ResearchTask(
                    id="t1",
                    description=f"fixed:{query}",
                    search_query=query,
                    providers=[SearchProviderHint.WEB],
                    priority=1,
                )
            ]
            return ResearchPlan(
                query=query,
                analysis=analysis,
                intent=QueryIntent.EXPLORATORY,
                tasks=tasks,
            )

    result = plan_node(initial_state("q"), planner=FixedPlanner())
    assert result["plan"] == ["fixed:q"]
    assert result["intent"] == QueryIntent.EXPLORATORY.value
    assert result["tasks"][0]["id"] == "t1"
    assert result["phase"] == "plan"


def test_plan_node_maps_validation_to_errors() -> None:
    result = plan_node(initial_state("  "))
    assert result["errors"] == ["query must not be empty"]
    assert result["phase"] == "failed"
