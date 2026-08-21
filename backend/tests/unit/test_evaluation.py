"""Tests for Phase 7.1 evaluation (RAGAS, DeepEval, heuristic metrics)."""

from __future__ import annotations

import pytest

from insightforge.core.config import Settings
from insightforge.core.exceptions import ExternalServiceError, ValidationFailedError
from insightforge.domain.models import EvaluationSample, RetrievalHit
from insightforge.graph import evaluate_node, initial_state
from insightforge.infrastructure.evaluation import (
    DeepEvalEvaluator,
    EvaluationService,
    HeuristicEvaluator,
    RagasEvaluator,
    append_evaluation_section,
    build_sample,
    create_evaluation_service,
    parse_evaluation_backend,
    resolve_backend,
)
from insightforge.shared.enums import EvaluationBackend, EvaluationMetric, RetrievalMode


def _sample(
    *,
    query: str = "RAG retrieves relevant context",
    answer: str = "RAG retrieves relevant context for grounded answers.",
    contexts: list[str] | None = None,
    ground_truth: str | None = None,
) -> EvaluationSample:
    if contexts is None:
        contexts = ["RAG retrieves relevant context for grounded answers from indexed documents."]
    return build_sample(
        query=query,
        answer=answer,
        contexts=contexts,
        ground_truth=ground_truth,
    )


def test_parse_evaluation_backend() -> None:
    assert parse_evaluation_backend("auto") is EvaluationBackend.AUTO
    assert parse_evaluation_backend("RAGAS") is EvaluationBackend.RAGAS
    assert parse_evaluation_backend("deepeval") is EvaluationBackend.DEEPEVAL
    assert parse_evaluation_backend("heuristic") is EvaluationBackend.HEURISTIC
    with pytest.raises(ValidationFailedError):
        parse_evaluation_backend("bleu")


def test_resolve_backend_auto_falls_back_to_heuristic() -> None:
    assert (
        resolve_backend(EvaluationBackend.AUTO, ragas_ok=False, deepeval_ok=False)
        is EvaluationBackend.HEURISTIC
    )
    assert resolve_backend(EvaluationBackend.RAGAS) is EvaluationBackend.RAGAS


def test_resolve_backend_auto_prefers_ragas() -> None:
    assert (
        resolve_backend(EvaluationBackend.AUTO, ragas_ok=True, deepeval_ok=True)
        is EvaluationBackend.RAGAS
    )
    assert (
        resolve_backend(EvaluationBackend.AUTO, ragas_ok=False, deepeval_ok=True)
        is EvaluationBackend.DEEPEVAL
    )


def test_heuristic_grounded_answer_scores_high() -> None:
    report = HeuristicEvaluator().evaluate(_sample())

    assert report.backend is EvaluationBackend.HEURISTIC
    assert report.context_count == 1
    assert report.score_for(EvaluationMetric.FAITHFULNESS) is not None
    assert report.score_for(EvaluationMetric.FAITHFULNESS) >= 0.5
    assert report.score_for(EvaluationMetric.RELEVANCY) >= 0.5
    assert report.score_for(EvaluationMetric.RECALL) >= 0.5
    assert report.score_for(EvaluationMetric.PRECISION) >= 0.5
    assert report.overall >= 0.5
    markdown = report.to_markdown()
    assert markdown.startswith("## Evaluation")
    assert "Faithfulness" in markdown


def test_heuristic_hallucinated_answer_has_low_faithfulness() -> None:
    grounded = HeuristicEvaluator().evaluate(_sample())
    hallucinated = HeuristicEvaluator().evaluate(
        _sample(answer="Cats sit on mats and chase red laser dots around the room.")
    )

    assert hallucinated.score_for(EvaluationMetric.FAITHFULNESS) < grounded.score_for(
        EvaluationMetric.FAITHFULNESS
    )
    assert hallucinated.score_for(EvaluationMetric.FAITHFULNESS) < 0.5


def test_heuristic_irrelevant_answer_has_low_relevancy() -> None:
    report = HeuristicEvaluator().evaluate(
        _sample(answer="The weather tomorrow is sunny with scattered clouds.")
    )

    assert report.score_for(EvaluationMetric.RELEVANCY) < 0.3


def test_heuristic_empty_contexts_zero_retrieval_metrics() -> None:
    report = HeuristicEvaluator().evaluate(_sample(contexts=[]))

    assert report.score_for(EvaluationMetric.FAITHFULNESS) == 0.0
    assert report.score_for(EvaluationMetric.RECALL) == 0.0
    assert report.score_for(EvaluationMetric.PRECISION) == 0.0


def test_heuristic_uses_ground_truth_for_recall() -> None:
    report = HeuristicEvaluator().evaluate(
        _sample(
            answer="Short answer.",
            ground_truth="RAG retrieves relevant context for grounded answers.",
        )
    )

    assert report.ground_truth_used is True
    assert report.score_for(EvaluationMetric.RECALL) >= 0.5
    reason = next(
        metric.reason for metric in report.metrics if metric.name is EvaluationMetric.RECALL
    )
    assert "ground truth" in reason


def test_heuristic_noisy_context_lowers_precision() -> None:
    clean = HeuristicEvaluator().evaluate(_sample())
    noisy = HeuristicEvaluator().evaluate(
        _sample(
            contexts=[
                "Cats sit on mats and ignore the question entirely.",
                "RAG retrieves relevant context for grounded answers from indexed documents.",
            ]
        )
    )

    assert noisy.score_for(EvaluationMetric.PRECISION) < clean.score_for(EvaluationMetric.PRECISION)


def test_append_evaluation_inserts_before_errors() -> None:
    report = HeuristicEvaluator().evaluate(_sample())
    markdown = "# Title\n\nBody.\n\n## Errors\n\n- search failed\n"
    combined = append_evaluation_section(markdown, report)

    eval_at = combined.index("## Evaluation")
    errors_at = combined.index("## Errors")
    assert eval_at < errors_at
    assert "search failed" in combined


def test_ragas_injected_run_fn() -> None:
    def fake_run(_sample: EvaluationSample) -> dict[str, float]:
        return {
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "context_recall": 0.7,
            "context_precision": 0.6,
        }

    report = RagasEvaluator(run_fn=fake_run).evaluate(_sample())

    assert report.backend is EvaluationBackend.RAGAS
    assert report.score_for(EvaluationMetric.FAITHFULNESS) == pytest.approx(0.9)
    assert report.score_for(EvaluationMetric.RELEVANCY) == pytest.approx(0.8)
    assert report.score_for(EvaluationMetric.RECALL) == pytest.approx(0.7)
    assert report.score_for(EvaluationMetric.PRECISION) == pytest.approx(0.6)
    assert report.overall == pytest.approx(0.75)


def test_deepeval_injected_run_fn() -> None:
    def fake_run(_sample: EvaluationSample) -> dict[str, float]:
        return {
            "faithfulness": 1.0,
            "relevancy": 0.5,
            "recall": 0.25,
            "precision": 0.0,
        }

    report = DeepEvalEvaluator(run_fn=fake_run).evaluate(_sample())

    assert report.backend is EvaluationBackend.DEEPEVAL
    assert report.overall == pytest.approx(0.4375)


def test_ragas_unavailable_without_llm() -> None:
    provider = RagasEvaluator()
    assert provider.available is False
    with pytest.raises(ExternalServiceError, match="ragas"):
        provider.evaluate(_sample())


def test_service_falls_back_when_ragas_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(_env_file=None, evaluation_backend="ragas")
    service = create_evaluation_service(settings)
    report = service.evaluate(_sample())

    assert report.backend is EvaluationBackend.HEURISTIC
    assert report.metadata.get("fallback_from") == "ragas"


def test_service_falls_back_when_primary_raises() -> None:
    def boom(_sample: EvaluationSample) -> dict[str, float]:
        raise RuntimeError("openai quota")

    service = EvaluationService(RagasEvaluator(run_fn=boom))
    report = service.evaluate(_sample())

    assert report.backend is EvaluationBackend.HEURISTIC
    assert report.metadata.get("fallback_from") == "ragas"


def test_service_rejects_empty_query() -> None:
    service = EvaluationService(HeuristicEvaluator())
    with pytest.raises(ValidationFailedError):
        service.evaluate(_sample(query="  ", answer="x"))


def test_create_service_from_settings() -> None:
    settings = Settings(_env_file=None, evaluation_backend="heuristic")
    service = create_evaluation_service(settings)

    assert service.enabled is True
    assert service.provider.name is EvaluationBackend.HEURISTIC
    assert service.available is True


def test_evaluate_node_appends_scores() -> None:
    state = initial_state("RAG retrieves relevant context")
    state["report"] = "# RAG\n\nFindings about retrieval.\n"
    state["reasoning"] = {
        "query": state["query"],
        "answer": "RAG retrieves relevant context for grounded answers.",
    }
    state["hits"] = [
        RetrievalHit(
            id="c0",
            score=0.9,
            text="RAG retrieves relevant context for grounded answers from indexed documents.",
            mode=RetrievalMode.BM25,
        ).model_dump(mode="json")
    ]
    state["phase"] = "report"

    result = evaluate_node(state, evaluator=EvaluationService(HeuristicEvaluator()))

    assert result["phase"] == "evaluate"
    assert result["transitions"] == ["report->evaluate"]
    assert "## Evaluation" in result["report"]
    assert result["evaluation"]["backend"] == "heuristic"
    assert result["evaluation"]["overall"] >= 0.0
    names = [item["name"] for item in result["evaluation"]["metrics"]]
    assert names == ["faithfulness", "relevancy", "recall", "precision"]


def test_evaluate_node_disabled_skips_scoring() -> None:
    settings = Settings(_env_file=None, evaluation_enabled=False)
    service = create_evaluation_service(settings)
    state = initial_state("topic")
    state["report"] = "# Topic\n"
    state["phase"] = "report"

    result = evaluate_node(state, evaluator=service)

    assert result["evaluation"] == {}
    assert result["phase"] == "evaluate"
    assert result["transitions"] == ["report->evaluate"]
