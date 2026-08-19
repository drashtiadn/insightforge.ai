"""Tests for agent interfaces (base, planner, tools) and Phase 6 agents."""

from __future__ import annotations

import pytest

from insightforge.agents import (
    BaseAgent,
    FunctionTool,
    Planner,
    ResearchPlan,
    ResearchTask,
    SimplePlanner,
    SimpleReasoner,
    SimpleReflectionAgent,
    SimpleReportGenerator,
    Tool,
)
from insightforge.agents.planner.schemas import QueryAnalysis
from insightforge.core.exceptions import ValidationFailedError
from insightforge.domain.models import (
    Document,
    ReasoningResult,
    ReflectionResult,
    ResearchReport,
    RetrievalHit,
)
from insightforge.graph.nodes import plan_node
from insightforge.graph.state import initial_state
from insightforge.shared.enums import (
    QueryIntent,
    RetrievalMode,
    SearchProviderHint,
)


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


# ---------------------------------------------------------------------------
# Phase 6 — Reasoner, Reflection, Report Generator
# ---------------------------------------------------------------------------


def _make_hit(hit_id: str, text: str, *, score: float = 0.9) -> RetrievalHit:
    return RetrievalHit(
        id=hit_id,
        score=score,
        text=text,
        metadata={"title": f"Doc {hit_id}", "url": f"https://example.com/{hit_id}"},
        mode=RetrievalMode.HYBRID,
    )


def _make_document(title: str, url: str, snippet: str) -> Document:
    return Document(
        title=title,
        url=url,
        snippet=snippet,
        provider=SearchProviderHint.WEB,
    )


def test_simple_reasoner_synthesizes_answer_from_hits() -> None:
    reasoner = SimpleReasoner()
    result = reasoner.reason(
        "vector databases",
        hits=[
            _make_hit("h1", "Vector databases store high-dimensional embeddings."),
            _make_hit("h2", "They power semantic search and RAG pipelines."),
        ],
    )

    assert isinstance(result, ReasoningResult)
    assert result.query == "vector databases"
    assert "Vector databases" in result.answer
    assert len(result.key_points) == 2
    assert result.used_source_ids == [
        "https://example.com/h1",
        "https://example.com/h2",
    ]
    assert 0.0 < result.confidence <= 0.9


def test_simple_reasoner_falls_back_to_documents_when_no_hits() -> None:
    reasoner = SimpleReasoner()
    doc = _make_document(
        "LangGraph",
        "https://example.com/langgraph",
        "LangGraph orchestrates stateful multi-agent workflows.",
    )
    result = reasoner.reason("multi agent frameworks", documents=[doc])

    assert result.key_points and "LangGraph" in result.key_points[0]
    assert result.used_source_ids == [doc.url]


def test_simple_reasoner_empty_evidence_returns_zero_confidence() -> None:
    result = SimpleReasoner().reason("orphan query")
    assert result.answer == ""
    assert result.key_points == []
    assert result.confidence == 0.0


def test_simple_reasoner_rejects_empty_query() -> None:
    with pytest.raises(ValidationFailedError):
        SimpleReasoner().reason("   ")


def test_simple_reflection_flags_insufficient_evidence() -> None:
    reasoning = ReasoningResult(
        query="edge cases",
        answer="short answer",
        key_points=["short answer"],
        used_source_ids=["h1"],
        confidence=0.3,
    )
    verdict = SimpleReflectionAgent().reflect(reasoning, hits=[_make_hit("h1", "x")])

    assert isinstance(verdict, ReflectionResult)
    assert verdict.is_sufficient is False
    assert verdict.gaps  # at least one gap recorded
    assert verdict.follow_up_queries


def test_simple_reflection_passes_when_evidence_is_strong() -> None:
    long_answer = "This is a sufficiently long synthesized answer. " * 5
    reasoning = ReasoningResult(
        query="strong evidence",
        answer=long_answer,
        key_points=["a", "b", "c"],
        used_source_ids=["h1", "h2", "h3"],
        confidence=0.8,
    )
    verdict = SimpleReflectionAgent().reflect(
        reasoning,
        hits=[_make_hit("h1", "x"), _make_hit("h2", "y"), _make_hit("h3", "z")],
    )

    assert verdict.is_sufficient is True
    assert verdict.gaps == []
    assert verdict.follow_up_queries == []


def test_simple_report_generator_builds_markdown_report() -> None:
    planner = SimplePlanner()
    plan = planner.build_plan("large language models")

    doc = _make_document(
        "LLM Survey",
        "https://example.com/llm-survey",
        "A survey of large language models and their applications.",
    )
    reasoning = SimpleReasoner().reason("large language models", documents=[doc])
    reflection = SimpleReflectionAgent().reflect(reasoning, documents=[doc])

    report = SimpleReportGenerator().generate(
        plan=plan,
        reasoning=reasoning,
        reflection=reflection,
        documents=[doc],
    )

    assert isinstance(report, ResearchReport)
    assert report.query == "large language models"
    assert report.title.startswith("Large language")
    assert any(section.heading == "Summary" for section in report.sections)
    assert any(section.heading == "Research Plan" for section in report.sections)
    assert any(section.heading == "Reflection" for section in report.sections)
    assert report.citations and report.citations[0].url == doc.url
    markdown = report.to_markdown()
    assert "# Large language models" in markdown
    assert "## Summary" in markdown
    assert "## Citations" in markdown
