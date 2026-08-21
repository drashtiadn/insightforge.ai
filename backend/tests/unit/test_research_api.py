"""Research HTTP API tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from insightforge.application.use_cases import ResearchRun, ResearchSourceRef, execute_research
from insightforge.core.config import get_settings
from insightforge.core.exceptions import ValidationFailedError
from insightforge.domain.models import EvaluationReport, MetricScore
from insightforge.graph import WorkflowResult, initial_state
from insightforge.shared.enums import EvaluationBackend, EvaluationMetric


def _run(**overrides: object) -> ResearchRun:
    defaults: dict[str, object] = {
        "query": "hybrid RAG",
        "report": "# Hybrid RAG\n\nFindings...",
        "score": 0.8,
        "confidence": 0.75,
        "phase": "done",
        "ok": True,
        "errors": (),
        "transitions": ("init->plan", "reflect->report"),
        "sources": (ResearchSourceRef(title="Paper", url="https://example.com/rag"),),
    }
    defaults.update(overrides)
    return ResearchRun(**defaults)  # type: ignore[arg-type]


def test_openapi_lists_research(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]

    assert "/api/v1/research" in paths
    assert "post" in paths["/api/v1/research"]


def test_research_returns_report(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_execute(
        query: str,
        *,
        max_steps: int | None = None,
        stub_search: bool = False,
    ) -> ResearchRun:
        captured["query"] = query
        captured["max_steps"] = max_steps
        captured["stub_search"] = stub_search
        return _run(query=query)

    monkeypatch.setattr("insightforge.api.routers.research.execute_research", fake_execute)

    response = client.post(
        "/api/v1/research",
        json={"query": "  hybrid RAG  ", "max_steps": 2, "stub_search": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "hybrid RAG"
    assert body["ok"] is True
    assert body["phase"] == "done"
    assert body["report"].startswith("# Hybrid RAG")
    assert body["sources"] == [{"title": "Paper", "url": "https://example.com/rag"}]
    assert body["evaluation"] is None
    assert body["judgment"] is None
    assert captured == {"query": "hybrid RAG", "max_steps": 2, "stub_search": True}


def test_research_returns_evaluation(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    evaluation = EvaluationReport(
        query="hybrid RAG",
        backend=EvaluationBackend.HEURISTIC,
        metrics=[
            MetricScore(name=EvaluationMetric.FAITHFULNESS, score=0.9, reason="grounded"),
            MetricScore(name=EvaluationMetric.RELEVANCY, score=0.8),
            MetricScore(name=EvaluationMetric.RECALL, score=0.7),
            MetricScore(name=EvaluationMetric.PRECISION, score=0.6),
        ],
        overall=0.75,
        context_count=2,
    )

    monkeypatch.setattr(
        "insightforge.api.routers.research.execute_research",
        lambda query, **kwargs: _run(query=query, evaluation=evaluation),
    )

    response = client.post("/api/v1/research", json={"query": "hybrid RAG", "stub_search": True})

    assert response.status_code == 200
    body = response.json()["evaluation"]
    assert body["backend"] == "heuristic"
    assert body["overall"] == pytest.approx(0.75)
    assert body["context_count"] == 2
    assert [item["name"] for item in body["metrics"]] == [
        "faithfulness",
        "relevancy",
        "recall",
        "precision",
    ]


def test_research_rejects_empty_query(client: TestClient) -> None:
    response = client.post("/api/v1/research", json={"query": "   "})

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation_failed"


def test_root_includes_research_url(client: TestClient) -> None:
    settings = get_settings()
    body = client.get("/").json()

    assert body["research_url"] == f"{settings.api_v1_prefix}/research"


def _workflow_result(**overrides: object) -> WorkflowResult:
    query = str(overrides.get("query", "hybrid RAG"))
    state = initial_state(query)
    state["sources"] = [
        {"title": "Paper", "url": "https://example.com/rag"},
        {"title": "Dup", "url": "https://example.com/rag"},
        {"title": "Other", "url": "https://example.com/other"},
    ]
    state["phase"] = "done"
    values: dict[str, object] = {
        "query": query,
        "report": "# Report",
        "score": 0.5,
        "confidence": 0.4,
        "phase": "done",
        "errors": (),
        "transitions": ("init->plan",),
        "state": state,
    }
    values.update(overrides)
    return WorkflowResult(**values)  # type: ignore[arg-type]


def test_execute_research_maps_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "insightforge.application.use_cases.research.run_research",
        lambda query, **kwargs: _workflow_result(query=query),
    )

    run = execute_research("hybrid RAG", max_steps=1, stub_search=True)

    assert run.ok
    assert [source.url for source in run.sources] == [
        "https://example.com/rag",
        "https://example.com/other",
    ]


def test_execute_research_maps_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    state = initial_state("hybrid RAG")
    state["sources"] = [{"title": "Paper", "url": "https://example.com/rag"}]
    state["evaluation"] = {
        "query": "hybrid RAG",
        "backend": "heuristic",
        "metrics": [
            {"name": "faithfulness", "score": 0.9, "reason": ""},
            {"name": "relevancy", "score": 0.8, "reason": ""},
            {"name": "recall", "score": 0.7, "reason": ""},
            {"name": "precision", "score": 0.6, "reason": ""},
        ],
        "overall": 0.75,
        "context_count": 3,
        "ground_truth_used": False,
        "metadata": {},
    }
    monkeypatch.setattr(
        "insightforge.application.use_cases.research.run_research",
        lambda query, **kwargs: _workflow_result(query=query, state=state),
    )

    run = execute_research("hybrid RAG", stub_search=True)

    assert run.evaluation is not None
    assert run.evaluation.backend is EvaluationBackend.HEURISTIC
    assert run.evaluation.overall == pytest.approx(0.75)
    assert run.evaluation.context_count == 3


def test_execute_research_maps_judgment(monkeypatch: pytest.MonkeyPatch) -> None:
    from insightforge.domain.models import JudgeVerdict

    state = initial_state("hybrid RAG")
    state["sources"] = [{"title": "Paper", "url": "https://example.com/rag"}]
    state["judgment"] = JudgeVerdict(
        passed=True,
        confidence=0.8,
        threshold=0.5,
        quality=0.75,
        backend="heuristic",
        attempt=0,
        max_retries=1,
    ).model_dump(mode="json")
    monkeypatch.setattr(
        "insightforge.application.use_cases.research.run_research",
        lambda query, **kwargs: _workflow_result(query=query, state=state),
    )

    run = execute_research("hybrid RAG", stub_search=True)

    assert run.judgment is not None
    assert run.judgment.passed is True
    assert run.judgment.confidence == pytest.approx(0.8)


def test_execute_research_blocks_stub_in_production(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(query: str, **kwargs: object) -> WorkflowResult:
        captured.update(kwargs)
        return _workflow_result(query=query)

    monkeypatch.setattr("insightforge.application.use_cases.research.run_research", fake_run)
    settings = MagicMock(is_production=True)

    execute_research("hybrid RAG", stub_search=True, settings=settings)

    assert captured["stub_search"] is False


def test_execute_research_rejects_empty_pipeline_query(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "insightforge.application.use_cases.research.run_research",
        lambda query, **kwargs: _workflow_result(
            query=query,
            phase="failed",
            errors=("query must not be empty",),
            report="",
            state=initial_state(query),
        ),
    )

    with pytest.raises(ValidationFailedError, match="Query must not be empty"):
        execute_research("hybrid RAG")
