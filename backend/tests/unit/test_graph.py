"""Tests for the LangGraph foundation."""

from __future__ import annotations

from insightforge.graph import (
    PASS_SCORE,
    after_evaluate,
    after_plan,
    compile_graph,
    evaluate_node,
    initial_state,
    plan_node,
    report_node,
    research_node,
    retrieve_node,
)


def test_plan_creates_steps() -> None:
    result = plan_node(initial_state("climate"))
    assert "errors" not in result
    assert len(result["plan"]) == 4


def test_plan_rejects_empty_query() -> None:
    result = plan_node(initial_state("  "))
    assert result["errors"] == ["query must not be empty"]


def test_research_and_retrieve() -> None:
    state = initial_state("AI")
    researched = research_node(state)
    assert researched["step"] == 1
    assert "AI" in researched["notes"][0]

    state["step"] = 1
    retrieved = retrieve_node(state)
    assert retrieved["sources"][0]["title"].startswith("Source 1")
    assert "example.com" in retrieved["sources"][0]["url"]


def test_evaluate_scores_notes_and_sources() -> None:
    state = initial_state("AI")
    state["notes"] = ["a"]
    state["sources"] = [{"title": "t", "url": "u"}]
    assert evaluate_node(state)["score"] == 0.6


def test_report_includes_query_and_sources() -> None:
    state = initial_state("AI")
    state["plan"] = ["step 1"]
    state["notes"] = ["note 1"]
    state["sources"] = [{"title": "Paper", "url": "https://example.com"}]
    state["score"] = 0.5
    report = report_node(state)["report"]
    assert "AI" in report
    assert "Paper" in report


def test_after_plan_routes() -> None:
    assert after_plan(initial_state("topic")) == "research"

    bad = initial_state("")
    bad["errors"] = ["query must not be empty"]
    assert after_plan(bad) == "__end__"


def test_after_evaluate_routes() -> None:
    state = initial_state("topic", max_steps=2)
    state["score"] = 0.2
    state["step"] = 1
    assert after_evaluate(state) == "research"

    state["score"] = PASS_SCORE
    assert after_evaluate(state) == "report"

    state["score"] = 0.1
    state["step"] = 2
    assert after_evaluate(state) == "report"


def test_full_graph_run() -> None:
    graph = compile_graph()
    result = graph.invoke(initial_state("multi-agent systems"))

    assert result["errors"] == []
    assert result["report"]
    assert "multi-agent systems" in result["report"]
    assert len(result["plan"]) == 4
    assert result["step"] == 2
    assert len(result["notes"]) == 2
    assert len(result["sources"]) == 2
    assert result["score"] >= PASS_SCORE


def test_full_graph_empty_query() -> None:
    graph = compile_graph()
    result = graph.invoke(initial_state("  "))

    assert result["errors"] == ["query must not be empty"]
    assert result["report"] == ""


def test_full_graph_stops_at_max_steps() -> None:
    graph = compile_graph()
    result = graph.invoke(initial_state("budget test", max_steps=1))

    assert result["step"] == 1
    assert len(result["notes"]) == 1
    assert len(result["sources"]) == 1
    # One pass scores 0.6 — below PASS_SCORE — but max_steps forces report.
    assert result["score"] < PASS_SCORE
    assert result["report"]
