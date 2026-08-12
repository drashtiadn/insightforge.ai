"""Tests for the LangGraph foundation."""

from __future__ import annotations

from insightforge.graph import (
    AUTO_MAX_STEPS,
    PASS_SCORE,
    after_evaluate,
    after_plan,
    compile_graph,
    evaluate_node,
    initial_state,
    plan_node,
    report_node,
    research_node,
    search_node,
)


def test_plan_creates_steps() -> None:
    result = plan_node(initial_state("climate"))
    assert "errors" not in result
    assert result["phase"] == "plan"
    assert result["transitions"] == ["init->plan"]
    assert result["intent"]
    assert len(result["plan"]) == 3
    assert len(result["tasks"]) == 3
    assert result["max_steps"] == 3


def test_plan_respects_explicit_max_steps_budget() -> None:
    result = plan_node(initial_state("climate", max_steps=1))
    assert result["max_steps"] == 1
    assert len(result["tasks"]) == 3


def test_plan_rejects_empty_query() -> None:
    result = plan_node(initial_state("  "))
    assert result["errors"] == ["query must not be empty"]
    assert result["phase"] == "failed"


def test_research_and_search() -> None:
    state = initial_state("AI")
    researched = research_node(state)
    assert researched["step"] == 1
    assert researched["phase"] == "research"
    assert "AI" in researched["notes"][0]

    state["step"] = 1
    state["phase"] = "research"
    searched = search_node(state)
    assert searched["phase"] == "search"
    assert searched["sources"][0]["title"].startswith("Source 1")
    assert "example.com" in searched["sources"][0]["url"]


def test_evaluate_scores_notes_and_sources() -> None:
    state = initial_state("AI")
    state["notes"] = ["a"]
    state["sources"] = [{"title": "t", "url": "u"}]
    result = evaluate_node(state)
    assert result["score"] == 0.6
    assert result["phase"] == "evaluate"


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
    bad["phase"] = "failed"
    assert after_plan(bad) == "__end__"


def test_after_evaluate_routes() -> None:
    state = initial_state("topic", max_steps=2)
    state["score"] = 0.2
    state["step"] = 1
    assert after_evaluate(state) == "research"

    state["score"] = PASS_SCORE
    state["tasks"] = []
    assert after_evaluate(state) == "report"

    state["score"] = 0.1
    state["step"] = 2
    assert after_evaluate(state) == "report"


def test_after_evaluate_continues_until_all_planned_tasks() -> None:
    state = initial_state("topic", max_steps=3)
    state["tasks"] = [{"id": "t1"}, {"id": "t2"}, {"id": "t3"}]
    state["score"] = 1.0
    state["step"] = 2
    # Score is high, but planned task t3 has not run yet.
    assert after_evaluate(state) == "research"

    state["step"] = 3
    assert after_evaluate(state) == "report"


def test_after_evaluate_recovers_with_partial_data() -> None:
    state = initial_state("topic")
    state["errors"] = ["search failed: boom"]
    state["notes"] = ["partial note"]
    assert after_evaluate(state) == "report"


def test_after_evaluate_ends_when_nothing_to_recover() -> None:
    state = initial_state("topic")
    state["errors"] = ["hard failure"]
    assert after_evaluate(state) == "__end__"


def test_full_graph_run() -> None:
    graph = compile_graph(stub_search=True)
    result = graph.invoke(initial_state("multi-agent systems"))

    assert result["errors"] == []
    assert result["phase"] == "done"
    assert result["report"]
    assert "multi-agent systems" in result["report"]
    assert result["intent"]
    assert len(result["plan"]) == 3
    assert len(result["tasks"]) == 3
    assert result["max_steps"] == 3
    assert result["step"] == 3
    assert len(result["notes"]) == 3
    assert len(result["sources"]) == 3
    assert result["score"] >= PASS_SCORE
    assert "init->plan" in result["transitions"]
    assert "research->search" in result["transitions"]
    assert result["transitions"][-1] == "evaluate->done"


def test_full_graph_empty_query() -> None:
    graph = compile_graph(stub_search=True)
    result = graph.invoke(initial_state("  "))

    assert result["errors"] == ["query must not be empty"]
    assert result["phase"] == "failed"
    assert result["report"] == ""


def test_full_graph_stops_at_max_steps() -> None:
    graph = compile_graph(stub_search=True)
    result = graph.invoke(initial_state("budget test", max_steps=1))

    assert result["step"] == 1
    assert len(result["notes"]) == 1
    assert len(result["sources"]) == 1
    # One pass scores 0.6 — below PASS_SCORE — but max_steps forces report.
    assert result["score"] < PASS_SCORE
    assert result["phase"] == "done"
    assert result["report"]


def test_initial_state_defaults_to_auto_max_steps() -> None:
    state = initial_state("q")
    assert state["max_steps"] == AUTO_MAX_STEPS


def test_full_graph_uses_injected_search_service() -> None:
    from insightforge.domain.models import Document
    from insightforge.infrastructure.search import SearchProvider, SearchService
    from insightforge.shared.enums import SearchProviderHint

    class _Provider(SearchProvider):
        name = SearchProviderHint.WIKIPEDIA

        def search(self, query: str, *, limit: int = 5) -> list[Document]:
            return [
                Document(
                    title=f"Doc for {query}",
                    url=f"https://example.com/{query.replace(' ', '-')}",
                    snippet=query,
                    content=query,
                    provider=SearchProviderHint.WIKIPEDIA,
                )
            ]

    service = SearchService(
        {SearchProviderHint.WIKIPEDIA: _Provider()},
        scoring=True,
    )
    # Planner tasks include wikipedia for exploratory queries.
    graph = compile_graph(search_service=service)
    result = graph.invoke(initial_state("photosynthesis", max_steps=1))

    assert result["errors"] == []
    assert result["phase"] == "done"
    assert len(result["documents"]) >= 1
    assert result["documents"][0]["title"].startswith("Doc for")
    assert len(result["sources"]) >= 1
    assert "example.com" in result["sources"][0]["url"]
