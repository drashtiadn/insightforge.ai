"""Tests for Phase 7.2 LLM-as-judge, retry, and confidence threshold."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from insightforge.agents.reflection.llm import LlmReflectionAgent
from insightforge.core.config import Settings
from insightforge.core.exceptions import ExternalServiceError
from insightforge.domain.models import EvaluationSample, ReasoningResult
from insightforge.graph import initial_state, judge_node
from insightforge.infrastructure.evaluation import (
    HeuristicJudge,
    JudgeService,
    LlmJudge,
    create_judge_service,
)
from insightforge.infrastructure.evaluation.sample import build_sample
from insightforge.infrastructure.llm.base import LlmClient, LlmMessage
from insightforge.infrastructure.llm.service import LlmService
from insightforge.shared.enums import LlmProviderHint


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


def _sample() -> EvaluationSample:
    return build_sample(
        query="RAG retrieves relevant context",
        answer="RAG retrieves relevant context for grounded answers.",
        contexts=["RAG retrieves relevant context for grounded answers from indexed documents."],
    )


def test_heuristic_judge_passes_above_threshold() -> None:
    from insightforge.infrastructure.evaluation.providers.heuristic import HeuristicEvaluator

    evaluation = HeuristicEvaluator().evaluate(_sample())
    verdict = HeuristicJudge().judge(
        _sample(),
        evaluation=evaluation,
        reflection_confidence=0.8,
        threshold=0.4,
        attempt=0,
        max_retries=1,
    )
    assert verdict.passed is True
    assert verdict.retry is False
    assert verdict.backend == "heuristic"
    assert "## Self-Reflection" in verdict.to_markdown()


def test_heuristic_judge_retries_below_threshold() -> None:
    verdict = HeuristicJudge().judge(
        _sample(),
        evaluation=None,
        reflection_confidence=0.1,
        threshold=0.5,
        attempt=0,
        max_retries=1,
    )
    assert verdict.passed is False
    assert verdict.retry is True
    assert verdict.revision_hint
    assert verdict.issues


def test_heuristic_judge_stops_when_retries_exhausted() -> None:
    verdict = HeuristicJudge().judge(
        _sample(),
        evaluation=None,
        reflection_confidence=0.1,
        threshold=0.9,
        attempt=1,
        max_retries=1,
    )
    assert verdict.passed is False
    assert verdict.retry is False


def test_llm_judge_parses_json() -> None:
    payload = '{"passed": true, "confidence": 0.9, "issues": [], "revision_hint": ""}'
    verdict = LlmJudge(LlmService(_FakeClient(payload))).judge(
        _sample(),
        evaluation=None,
        reflection_confidence=0.8,
        threshold=0.5,
        attempt=0,
        max_retries=1,
    )
    assert verdict.backend == "llm"
    assert verdict.passed is True
    assert verdict.confidence == pytest.approx(0.9)


def test_llm_judge_falls_back_on_bad_json() -> None:
    verdict = LlmJudge(LlmService(_FakeClient("not-json"))).judge(
        _sample(),
        evaluation=None,
        reflection_confidence=0.8,
        threshold=0.5,
        attempt=0,
        max_retries=1,
    )
    assert verdict.backend == "heuristic"
    assert verdict.passed is True


def test_llm_judge_rejects_low_confidence_even_if_passed_flag() -> None:
    payload = (
        '{"passed": true, "confidence": 0.2, "issues": ["thin"], "revision_hint": "add evidence"}'
    )
    verdict = LlmJudge(LlmService(_FakeClient(payload))).judge(
        _sample(),
        evaluation=None,
        reflection_confidence=0.2,
        threshold=0.5,
        attempt=0,
        max_retries=1,
    )
    assert verdict.passed is False
    assert verdict.retry is True
    assert "add evidence" in verdict.revision_hint


def test_create_judge_service_heuristic_for_stub() -> None:
    settings = Settings(_env_file=None)
    service = create_judge_service(settings, heuristic=True)
    verdict = service.judge(_sample(), reflection_confidence=0.8)
    assert service.enabled is True
    assert service.threshold == pytest.approx(0.5)
    assert verdict.backend == "heuristic"


def test_judge_node_appends_verdict() -> None:
    state = initial_state("RAG retrieves relevant context")
    state["report"] = "# RAG\n\nFindings.\n"
    state["reasoning"] = {
        "query": state["query"],
        "answer": "RAG retrieves relevant context for grounded answers.",
    }
    state["phase"] = "evaluate"
    service = JudgeService(HeuristicJudge(), threshold=0.0, max_retries=1)
    result = judge_node(state, judge=service)

    assert result["phase"] == "done"
    assert result["transitions"] == ["evaluate->done"]
    assert result["judgment"]["passed"] is True
    assert "## Self-Reflection" in result["report"]


def test_judge_node_retries_reason() -> None:
    state = initial_state("thin")
    state["report"] = "# Thin\n"
    state["phase"] = "evaluate"
    service = JudgeService(HeuristicJudge(), threshold=0.99, max_retries=1)
    result = judge_node(state, judge=service)

    assert result["phase"] == "judge"
    assert result["transitions"] == ["evaluate->judge"]
    assert result["judgment"]["retry"] is True
    assert result["judge_retries"] == 1
    assert result["revision_hint"]


def test_judge_node_disabled_skips() -> None:
    settings = Settings(_env_file=None, judge_enabled=False)
    service = create_judge_service(settings, heuristic=True)
    state = initial_state("topic")
    state["phase"] = "evaluate"
    result = judge_node(state, judge=service)
    assert result["judgment"] == {}
    assert result["phase"] == "done"


def test_llm_reflection_parses_json() -> None:
    payload = (
        '{"is_sufficient": true, "gaps": [], "follow_up_queries": [],'
        ' "reasoning": "enough evidence", "confidence": 0.8}'
    )
    agent = LlmReflectionAgent(LlmService(_FakeClient(payload)))
    reasoning = ReasoningResult(
        query="RAG",
        answer="Retrieval augmented generation retrieves documents then generates an answer.",
        key_points=["retrieve", "generate"],
        used_source_ids=["a", "b"],
        confidence=0.8,
    )
    verdict = agent.reflect(reasoning)
    assert verdict.is_sufficient is True
    assert verdict.confidence == pytest.approx(0.8)
    assert "enough evidence" in verdict.reasoning


def test_simple_reasoner_records_revision_feedback() -> None:
    from insightforge.agents import SimpleReasoner
    from insightforge.domain.models import Document
    from insightforge.shared.enums import SearchProviderHint

    docs = [
        Document(
            title="RAG",
            url="https://example.com/rag",
            snippet="Retrieval augmented generation retrieves documents then generates.",
            content="Retrieval augmented generation retrieves documents then generates an answer.",
            provider=SearchProviderHint.WEB,
        )
    ]
    result = SimpleReasoner().reason(
        "what is RAG",
        documents=docs,
        feedback="Ground claims in the retrieved context.",
    )
    assert result.metadata["revision_hint"] == "Ground claims in the retrieved context."


def test_llm_reflection_falls_back_on_bad_json() -> None:
    agent = LlmReflectionAgent(LlmService(_FakeClient("nope")))
    reasoning = ReasoningResult(query="RAG", answer="short", confidence=0.2)
    verdict = agent.reflect(reasoning)
    assert verdict.is_sufficient is False
    assert verdict.gaps
