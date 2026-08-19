"""Tests for the LangGraph foundation and Phase 6 pipeline."""

from __future__ import annotations

from insightforge.graph import (
    AUTO_MAX_STEPS,
    after_plan,
    after_reflect,
    after_search,
    compile_graph,
    ingest_node,
    initial_state,
    plan_node,
    report_node,
    research_node,
    retrieve_node,
    search_node,
)
from insightforge.infrastructure.retrieval import RetrievalService
from insightforge.infrastructure.vectorstores.stores import MemoryVectorStore


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
    assert searched["documents"][0]["snippet"]


def test_report_includes_query() -> None:
    state = initial_state("AI")
    state["plan"] = ["step 1"]
    state["intent"] = "exploratory"
    state["documents"] = [
        {
            "title": "Paper",
            "url": "https://example.com",
            "snippet": "AI systems retrieve context.",
            "content": "AI systems retrieve context for answers.",
            "provider": "web",
        }
    ]
    from insightforge.graph.pipeline import reason_node

    reasoned = reason_node(state)
    state["reasoning"] = reasoned["reasoning"]
    report = report_node(state)["report"]
    assert "AI" in report
    assert report.startswith("#")


def test_after_plan_routes() -> None:
    assert after_plan(initial_state("topic")) == "research"

    bad = initial_state("")
    bad["errors"] = ["query must not be empty"]
    bad["phase"] = "failed"
    assert after_plan(bad) == "__end__"


def test_after_search_routes() -> None:
    state = initial_state("topic", max_steps=2)
    state["step"] = 1
    state["tasks"] = [{"id": "t1"}, {"id": "t2"}]
    assert after_search(state) == "research"

    state["step"] = 2
    assert after_search(state) == "ingest"

    state["tasks"] = [{"id": "t1"}, {"id": "t2"}, {"id": "t3"}]
    state["step"] = 2
    state["max_steps"] = 3
    assert after_search(state) == "research"

    state["step"] = 3
    assert after_search(state) == "ingest"


def test_after_search_recovers_with_partial_data() -> None:
    state = initial_state("topic", max_steps=1)
    state["errors"] = ["search failed: boom"]
    state["notes"] = ["partial note"]
    state["step"] = 1
    assert after_search(state) == "ingest"


def test_after_search_ends_when_nothing_to_recover() -> None:
    state = initial_state("topic")
    state["errors"] = ["hard failure"]
    assert after_search(state) == "__end__"


def test_after_reflect_reports_by_default() -> None:
    state = initial_state("topic", max_steps=1)
    state["step"] = 1
    assert after_reflect(state) == "report"


def test_ingest_and_retrieve_bm25() -> None:
    state = initial_state("retrieval augmented generation")
    state["documents"] = [
        {
            "title": "RAG intro",
            "url": "https://example.com/rag",
            "snippet": "Retrieval augmented generation combines search with LLMs.",
            "content": "Retrieval augmented generation combines search with LLMs.",
            "provider": "web",
        }
    ]
    retrieval = RetrievalService(MemoryVectorStore())
    ingested = ingest_node(state, retrieval=retrieval)
    assert ingested["phase"] == "ingest"
    assert ingested["chunks"]
    state["chunks"] = ingested["chunks"]
    retrieved = retrieve_node(state, retrieval=retrieval)
    assert retrieved["hits"]
    assert retrieved["hits"][0]["text"]


def test_full_graph_run() -> None:
    graph = compile_graph(stub_search=True)
    result = graph.invoke(initial_state("multi-agent systems"))

    assert result["errors"] == []
    assert result["phase"] == "done"
    assert result["report"]
    assert "multi-agent systems" in result["report"].lower()
    assert result["intent"]
    assert len(result["plan"]) == 3
    assert len(result["tasks"]) >= 3
    assert result["max_steps"] >= 3
    assert result["step"] >= 3
    assert len(result["notes"]) >= 3
    assert len(result["sources"]) >= 3
    assert "init->plan" in result["transitions"]
    assert "research->search" in result["transitions"]
    assert "search->ingest" in result["transitions"] or any(
        item.endswith("->ingest") for item in result["transitions"]
    )
    assert "retrieve->reason" in result["transitions"]
    assert "reason->reflect" in result["transitions"]
    assert result["transitions"][-1] == "report->done"
    assert "##" in result["report"]
    assert "## Evaluation" in result["report"]
    assert result["evaluation"]
    assert result["evaluation"]["metrics"]


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
    assert result["phase"] == "done"
    assert result["report"]
    assert "budget test" in result["report"].lower()


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
                    snippet=f"Long-form background about {query} with enough detail.",
                    content=f"Long-form background about {query} with enough detail for reasoning.",
                    provider=SearchProviderHint.WIKIPEDIA,
                )
            ]

    service = SearchService(
        {SearchProviderHint.WIKIPEDIA: _Provider()},
        scoring=True,
    )
    from insightforge.graph.workflow import assemble_resources

    resources = assemble_resources(
        search_service=service,
        retrieval=RetrievalService(MemoryVectorStore()),
    )
    graph = compile_graph(resources=resources)
    result = graph.invoke(initial_state("photosynthesis", max_steps=1))

    assert result["errors"] == []
    assert result["phase"] == "done"
    assert len(result["documents"]) >= 1
    assert result["documents"][0]["title"].startswith("Doc for")
    assert len(result["sources"]) >= 1
    assert "example.com" in result["sources"][0]["url"]
