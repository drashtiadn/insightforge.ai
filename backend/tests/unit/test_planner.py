"""Tests for Phase 3 planner: analysis, intent, and task decomposition."""

from __future__ import annotations

import pytest

from insightforge.agents.planner.analysis import (
    analyze_query,
    detect_intent,
    extract_keywords,
    normalize_query,
)
from insightforge.agents.planner.decompose import decompose_tasks
from insightforge.agents.planner.simple import SimplePlanner
from insightforge.core.exceptions import ValidationFailedError
from insightforge.graph.nodes import plan_node
from insightforge.graph.pipeline import report_node
from insightforge.graph.state import initial_state
from insightforge.shared.enums import QueryIntent, SearchProviderHint


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  hello   world  ", "hello world"),
        ("AI", "AI"),
    ],
)
def test_normalize_query(raw: str, expected: str) -> None:
    assert normalize_query(raw) == expected


def test_extract_keywords_drops_stopwords() -> None:
    assert extract_keywords("what is climate change") == ["climate", "change"]


@pytest.mark.parametrize(
    ("query", "intent"),
    [
        ("what is quantum computing", QueryIntent.FACTUAL),
        ("compare React vs Vue", QueryIntent.COMPARATIVE),
        ("how to train a transformer", QueryIntent.HOWTO),
        ("latest news on AI regulation", QueryIntent.NEWS),
        ("arxiv papers on diffusion models", QueryIntent.ACADEMIC),
        ("github library for vector search", QueryIntent.CODE),
        ("multi-agent research systems", QueryIntent.EXPLORATORY),
    ],
)
def test_detect_intent(query: str, intent: QueryIntent) -> None:
    assert detect_intent(normalize_query(query)) == intent


def test_analyze_query_builds_structure() -> None:
    analysis = analyze_query("  What is RAG?  ")
    assert analysis.original_query == "  What is RAG?  "
    assert analysis.normalized_query == "What is RAG?"
    assert "rag" in analysis.keywords
    assert analysis.token_count >= 2


def test_decompose_factual_includes_wikipedia() -> None:
    analysis = analyze_query("what is photosynthesis")
    tasks = decompose_tasks(analysis, QueryIntent.FACTUAL)

    assert len(tasks) == 3
    assert tasks[0].priority == 1
    assert SearchProviderHint.WIKIPEDIA in tasks[0].providers
    assert all(task.search_query for task in tasks)


def test_decompose_academic_prefers_arxiv() -> None:
    analysis = analyze_query("arxiv papers on transformers")
    tasks = decompose_tasks(analysis, QueryIntent.ACADEMIC)

    assert SearchProviderHint.ARXIV in tasks[0].providers


def test_decompose_code_prefers_github() -> None:
    analysis = analyze_query("github library for embeddings")
    tasks = decompose_tasks(analysis, QueryIntent.CODE)

    assert SearchProviderHint.GITHUB in tasks[0].providers


def test_simple_planner_build_plan() -> None:
    plan = SimplePlanner().build_plan("how to deploy FastAPI")

    assert plan.intent == QueryIntent.HOWTO
    assert len(plan.tasks) == 3
    assert plan.steps == [task.description for task in plan.tasks]
    assert plan.analysis.normalized_query == "how to deploy FastAPI"


def test_simple_planner_plan_matches_steps() -> None:
    planner = SimplePlanner()
    query = "compare PostgreSQL vs MySQL"
    assert planner.plan(query) == planner.build_plan(query).steps


def test_simple_planner_run_returns_research_plan() -> None:
    plan = SimplePlanner().run("what is LangGraph")
    assert plan.intent == QueryIntent.FACTUAL
    assert plan.tasks


def test_simple_planner_rejects_empty_query() -> None:
    with pytest.raises(ValidationFailedError) as exc_info:
        SimplePlanner().build_plan("   ")

    assert exc_info.value.message == "query must not be empty"
    assert exc_info.value.details == {"field": "query"}


def test_plan_node_stores_intent_and_tasks() -> None:
    result = plan_node(initial_state("what is reinforcement learning"))

    assert result["phase"] == "plan"
    assert result["intent"] == QueryIntent.FACTUAL.value
    assert len(result["plan"]) == 3
    assert len(result["tasks"]) == 3
    assert result["tasks"][0]["id"] == "t1"
    assert "wikipedia" in result["tasks"][0]["providers"]


def test_report_includes_intent() -> None:
    state = initial_state("AI")
    state["intent"] = QueryIntent.EXPLORATORY.value
    state["plan"] = ["overview"]
    state["notes"] = ["note"]
    state["sources"] = [{"title": "Paper", "url": "https://example.com"}]
    state["score"] = 0.5

    report = report_node(state)["report"]
    assert "AI" in report
    assert "Paper" in report
    assert report.startswith("#")
