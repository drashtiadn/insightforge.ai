"""Evaluation service — select RAGAS, DeepEval, or heuristic scoring."""

from __future__ import annotations

import os

from insightforge.core.config import Settings, get_settings
from insightforge.core.exceptions import ValidationFailedError
from insightforge.core.logging import get_logger
from insightforge.domain.models import EvaluationReport, EvaluationSample
from insightforge.infrastructure.evaluation.base import Evaluator
from insightforge.infrastructure.evaluation.providers import (
    DeepEvalEvaluator,
    DeepEvalRunFn,
    HeuristicEvaluator,
    RagasEvaluator,
    RagasRunFn,
    deepeval_installed,
    ragas_installed,
)
from insightforge.shared.enums import EvaluationBackend

logger = get_logger(__name__)


def parse_evaluation_backend(value: str) -> EvaluationBackend:
    """Parse settings/env value into ``EvaluationBackend``."""

    cleaned = value.strip().lower()
    try:
        return EvaluationBackend(cleaned)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in EvaluationBackend)
        raise ValidationFailedError(
            f"unknown evaluation backend {value!r}; expected one of: {allowed}",
            details={"field": "evaluation_backend", "value": value},
        ) from exc


def openai_configured(settings: Settings | None = None) -> bool:
    """True when an OpenAI key is available for RAGAS / DeepEval judges."""

    if os.environ.get("OPENAI_API_KEY"):
        return True
    cfg = settings
    return cfg is not None and cfg.openai_api_key is not None


def resolve_backend(
    hint: EvaluationBackend,
    *,
    ragas_ok: bool = False,
    deepeval_ok: bool = False,
) -> EvaluationBackend:
    """Pick a concrete backend for ``auto``, otherwise return ``hint``."""

    if hint is not EvaluationBackend.AUTO:
        return hint
    if ragas_ok:
        logger.info("evaluation auto-selected backend=ragas")
        return EvaluationBackend.RAGAS
    if deepeval_ok:
        logger.info("evaluation auto-selected backend=deepeval")
        return EvaluationBackend.DEEPEVAL
    logger.info("evaluation auto-selected backend=heuristic")
    return EvaluationBackend.HEURISTIC


def build_provider(
    settings: Settings,
    *,
    ragas_run_fn: RagasRunFn | None = None,
    deepeval_run_fn: DeepEvalRunFn | None = None,
) -> Evaluator:
    """Construct the configured evaluation backend."""

    hint = parse_evaluation_backend(settings.evaluation_backend)
    llm_ok = openai_configured(settings)
    ragas_ok = ragas_run_fn is not None or (ragas_installed() and llm_ok)
    deepeval_ok = deepeval_run_fn is not None or (deepeval_installed() and llm_ok)
    chosen = resolve_backend(hint, ragas_ok=ragas_ok, deepeval_ok=deepeval_ok)
    if chosen is EvaluationBackend.RAGAS:
        return RagasEvaluator(run_fn=ragas_run_fn, llm_configured=llm_ok)
    if chosen is EvaluationBackend.DEEPEVAL:
        return DeepEvalEvaluator(run_fn=deepeval_run_fn, llm_configured=llm_ok)
    return HeuristicEvaluator()


class EvaluationService:
    """Facade over one evaluation backend with heuristic fallback.

    Live research runs call ``evaluate`` after the report is generated.
    RAGAS and DeepEval failures fall back to the lexical scorer so a report
    always receives quality scores when evaluation is enabled.
    """

    def __init__(
        self,
        provider: Evaluator,
        *,
        enabled: bool = True,
        append_to_report: bool = True,
        fallback: Evaluator | None = None,
    ) -> None:
        self._provider = provider
        self._enabled = enabled
        self._append_to_report = append_to_report
        self._fallback = fallback or HeuristicEvaluator()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def append_to_report(self) -> bool:
        return self._append_to_report

    @property
    def provider(self) -> Evaluator:
        return self._provider

    @property
    def available(self) -> bool:
        return self._provider.available or self._fallback.available

    def evaluate(self, sample: EvaluationSample) -> EvaluationReport:
        """Score ``sample``; fall back to heuristic when the primary backend fails."""

        if not sample.query.strip():
            raise ValidationFailedError(
                "evaluation query must not be empty",
                details={"field": "query"},
            )

        logger.info(
            "evaluation started backend=%s available=%s query=%r contexts=%d",
            self._provider.name.value,
            self._provider.available,
            sample.query,
            len(sample.contexts),
            extra={
                "backend": self._provider.name.value,
                "available": self._provider.available,
                "contexts": len(sample.contexts),
                "has_ground_truth": sample.ground_truth is not None,
            },
        )

        report = self._run_with_fallback(sample)
        metric_scores = {item.name.value: round(item.score, 3) for item in report.metrics}
        logger.info(
            "evaluation finished backend=%s overall=%.3f metrics=%s",
            report.backend.value,
            report.overall,
            metric_scores,
            extra={
                "backend": report.backend.value,
                "overall": report.overall,
                "metrics": metric_scores,
            },
        )
        return report

    def _run_with_fallback(self, sample: EvaluationSample) -> EvaluationReport:
        if self._provider.available:
            try:
                return self._provider.evaluate(sample)
            except Exception as exc:
                if self._provider.name is EvaluationBackend.HEURISTIC:
                    raise
                logger.warning(
                    "evaluation backend=%s failed error=%s; falling back to heuristic",
                    self._provider.name.value,
                    exc,
                    extra={"backend": self._provider.name.value, "error": str(exc)},
                )
        elif self._provider.name is not EvaluationBackend.HEURISTIC:
            logger.warning(
                "evaluation backend=%s unavailable; falling back to heuristic",
                self._provider.name.value,
                extra={"backend": self._provider.name.value},
            )

        report = self._fallback.evaluate(sample)
        if self._provider.name is not EvaluationBackend.HEURISTIC:
            report.metadata["fallback_from"] = self._provider.name.value
        return report


def create_evaluation_service(
    settings: Settings | None = None,
    *,
    ragas_run_fn: RagasRunFn | None = None,
    deepeval_run_fn: DeepEvalRunFn | None = None,
) -> EvaluationService:
    """Factory used by the graph and tests."""

    cfg = settings or get_settings()
    provider = build_provider(
        cfg,
        ragas_run_fn=ragas_run_fn,
        deepeval_run_fn=deepeval_run_fn,
    )
    logger.info(
        "evaluation service created backend=%s enabled=%s available=%s append=%s",
        provider.name.value,
        cfg.evaluation_enabled,
        provider.available,
        cfg.evaluation_append_to_report,
        extra={
            "backend": provider.name.value,
            "enabled": cfg.evaluation_enabled,
            "available": provider.available,
        },
    )
    return EvaluationService(
        provider,
        enabled=cfg.evaluation_enabled,
        append_to_report=cfg.evaluation_append_to_report,
    )
