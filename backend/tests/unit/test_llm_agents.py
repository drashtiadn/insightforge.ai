"""Tests for Gemini-backed agents and heuristic fallback."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from insightforge.agents import (
    LlmPlanner,
    LlmReasoner,
    LlmReportGenerator,
    SimplePlanner,
    create_planner,
)
from insightforge.agents.planner.schemas import ResearchPlan
from insightforge.core.exceptions import ExternalServiceError, ValidationFailedError
from insightforge.domain.models import Document, ReasoningResult
from insightforge.infrastructure.llm.base import LlmClient, LlmMessage
from insightforge.infrastructure.llm.service import LlmService
from insightforge.shared.enums import LlmProviderHint, SearchProviderHint


class _FakeClient(LlmClient):
    name = LlmProviderHint.GEMINI

    def __init__(self, text: str, *, available: bool = True) -> None:
        self._text = text
        self._available = available
        self.calls: list[list[LlmMessage]] = []

    @property
    def available(self) -> bool:
        return self._available

    @property
    def model(self) -> str:
        return "fake"

    def complete(
        self,
        messages: Sequence[LlmMessage],
        *,
        json_mode: bool = False,
    ) -> str:
        self.calls.append(list(messages))
        if not self._available:
            raise ExternalServiceError("unavailable")
        return self._text


def test_llm_planner_parses_json_plan() -> None:
    payload = """
    {"intent": "academic",
     "tasks": [
       {"id": "t1", "description": "Find papers", "search_query": "RAG papers",
        "providers": ["arxiv", "web"], "priority": 1}
     ]}
    """
    planner = LlmPlanner(LlmService(_FakeClient(payload)))
    plan = planner.build_plan("RAG evaluation")
    assert plan.intent.value == "academic"
    assert plan.tasks[0].providers[0] is SearchProviderHint.ARXIV


def test_llm_planner_falls_back_on_invalid_json() -> None:
    planner = LlmPlanner(LlmService(_FakeClient("not-json")))
    plan = planner.build_plan("climate models")
    assert isinstance(plan, ResearchPlan)
    assert len(plan.tasks) == 3


def test_llm_planner_unavailable_uses_simple() -> None:
    planner = LlmPlanner(LlmService(_FakeClient("{}", available=False)))
    plan = planner.build_plan("how to train lora")
    assert plan.tasks
    assert create_planner(LlmService(_FakeClient("{}", available=False))).name == "planner"


def test_llm_planner_rejects_empty_query() -> None:
    planner = LlmPlanner(LlmService(_FakeClient("{}")))
    with pytest.raises(ValidationFailedError):
        planner.build_plan("  ")


def test_llm_reasoner_overrides_answer() -> None:
    payload = (
        '{"answer": "RAG retrieves then generates.", "key_points": ["retrieve"], "confidence": 0.8}'
    )
    reasoner = LlmReasoner(LlmService(_FakeClient(payload)))
    docs = [
        Document(
            title="RAG",
            url="https://example.com/rag",
            snippet="Retrieval augmented generation retrieves documents then generates.",
            content="Retrieval augmented generation retrieves documents then generates an answer.",
            provider=SearchProviderHint.WEB,
        )
    ]
    result = reasoner.reason("what is RAG", documents=docs)
    assert "retrieves then generates" in result.answer
    assert result.confidence == 0.8
    assert result.clusters


def test_llm_reasoner_falls_back_on_bad_json() -> None:
    reasoner = LlmReasoner(LlmService(_FakeClient("oops")))
    docs = [
        Document(
            title="RAG",
            url="https://example.com/rag",
            snippet="Retrieval augmented generation retrieves documents then generates.",
            content="Retrieval augmented generation retrieves documents then generates an answer.",
            provider=SearchProviderHint.WEB,
        )
    ]
    result = reasoner.reason("what is RAG", documents=docs)
    assert isinstance(result, ReasoningResult)
    assert result.answer


def test_llm_report_uses_llm_summary() -> None:
    payload = '{"title": "RAG Report", "summary": "RAG helps.", "findings": ["It retrieves"]}'
    reporter = LlmReportGenerator(LlmService(_FakeClient(payload)))
    planner = SimplePlanner()
    plan = planner.build_plan("RAG systems")
    reasoning = ReasoningResult(
        query="RAG systems",
        answer="RAG retrieves then generates.",
    )
    report = reporter.generate(plan=plan, reasoning=reasoning)
    assert report.title == "RAG Report"
    assert "RAG helps." in report.summary
    assert any(section.heading == "Key Findings" for section in report.sections)
