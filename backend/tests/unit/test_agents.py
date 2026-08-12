"""Tests for agent interfaces (base, planner, tools)."""

from __future__ import annotations

import pytest

from insightforge.agents import (
    BaseAgent,
    FunctionTool,
    Planner,
    SimplePlanner,
    Tool,
)
from insightforge.core.exceptions import ValidationFailedError
from insightforge.graph.nodes import plan_node
from insightforge.graph.state import initial_state


def test_simple_planner_builds_steps() -> None:
    planner = SimplePlanner()
    steps = planner.plan("climate models")

    assert planner.name == "planner"
    assert isinstance(planner, BaseAgent)
    assert isinstance(planner, Planner)
    assert len(steps) == 4
    assert "climate models" in steps[0]


def test_simple_planner_run_matches_plan() -> None:
    planner = SimplePlanner()
    assert planner.run("AI safety") == planner.plan("AI safety")


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
    assert result["plan"][0] == "Research: multi-agent systems"


def test_plan_node_accepts_custom_planner() -> None:
    class FixedPlanner(Planner):
        def plan(self, query: str) -> list[str]:
            return [f"fixed:{query}"]

    result = plan_node(initial_state("q"), planner=FixedPlanner())
    assert result["plan"] == ["fixed:q"]
    assert result["phase"] == "plan"


def test_plan_node_maps_validation_to_errors() -> None:
    result = plan_node(initial_state("  "))
    assert result["errors"] == ["query must not be empty"]
    assert result["phase"] == "failed"
