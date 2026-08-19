"""DeepEval adapter (optional extra: ``uv sync --extra evaluation``)."""

from __future__ import annotations

import importlib.util
import warnings
from collections.abc import Callable, Mapping
from typing import Any

from insightforge.core.exceptions import ExternalServiceError
from insightforge.core.logging import get_logger
from insightforge.domain.models import EvaluationReport, EvaluationSample
from insightforge.infrastructure.evaluation.base import Evaluator
from insightforge.infrastructure.evaluation.metrics import build_report
from insightforge.shared.enums import EvaluationBackend, EvaluationMetric

logger = get_logger(__name__)

# Injected scorer: sample in → metric-name → score. Used by unit tests.
DeepEvalRunFn = Callable[[EvaluationSample], Mapping[str, float]]


def deepeval_installed() -> bool:
    """True when the ``deepeval`` package is importable (does not import it)."""

    return importlib.util.find_spec("deepeval") is not None


class DeepEvalEvaluator(Evaluator):
    """Score a sample with DeepEval, or an injected function in tests."""

    name = EvaluationBackend.DEEPEVAL

    def __init__(
        self,
        *,
        run_fn: DeepEvalRunFn | None = None,
        llm_configured: bool = False,
    ) -> None:
        self._run_fn = run_fn
        self._llm_configured = llm_configured

    @property
    def available(self) -> bool:
        if self._run_fn is not None:
            return True
        return deepeval_installed() and self._llm_configured

    def evaluate(self, sample: EvaluationSample) -> EvaluationReport:
        if self._run_fn is None and not deepeval_installed():
            raise ExternalServiceError(
                "deepeval is not installed; install with: uv sync --extra evaluation",
                details={"backend": self.name.value},
            )
        if self._run_fn is None and not self._llm_configured:
            raise ExternalServiceError(
                "deepeval requires OPENAI_API_KEY",
                details={"backend": self.name.value},
            )

        logger.info(
            "deepeval evaluation started query=%r contexts=%d",
            sample.query,
            len(sample.contexts),
            extra={"backend": self.name.value, "contexts": len(sample.contexts)},
        )
        try:
            scores = dict(
                self._run_fn(sample) if self._run_fn is not None else _run_deepeval(sample)
            )
        except ExternalServiceError:
            raise
        except Exception as exc:
            logger.exception("deepeval evaluation failed query=%r", sample.query)
            raise ExternalServiceError(
                f"deepeval evaluation failed: {exc}",
                details={"backend": self.name.value},
            ) from exc

        logger.info(
            "deepeval evaluation finished query=%r scores=%s",
            sample.query,
            scores,
            extra={"backend": self.name.value, "scores": dict(scores)},
        )
        return build_report(sample, backend=self.name, scores=scores)


def _run_deepeval(sample: EvaluationSample) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            from deepeval.metrics import (
                AnswerRelevancyMetric,
                ContextualPrecisionMetric,
                ContextualRecallMetric,
                FaithfulnessMetric,
            )
            from deepeval.test_case import LLMTestCase
        except ImportError as exc:
            raise ExternalServiceError(
                "deepeval is not installed; install with: uv sync --extra evaluation",
                details={"backend": EvaluationBackend.DEEPEVAL.value},
            ) from exc

        case = LLMTestCase(
            input=sample.query,
            actual_output=sample.answer,
            retrieval_context=list(sample.contexts),
            expected_output=sample.reference,
        )
        specs: list[tuple[EvaluationMetric, Any]] = [
            (EvaluationMetric.FAITHFULNESS, FaithfulnessMetric()),
            (EvaluationMetric.RELEVANCY, AnswerRelevancyMetric()),
            (EvaluationMetric.RECALL, ContextualRecallMetric()),
            (EvaluationMetric.PRECISION, ContextualPrecisionMetric()),
        ]
        scores: dict[str, float] = {}
        for name, metric in specs:
            metric.measure(case)
            raw = getattr(metric, "score", None)
            if raw is None:
                logger.warning("deepeval metric=%s returned no score", name.value)
                continue
            scores[name.value] = float(raw)
        if not scores:
            raise ExternalServiceError(
                "deepeval returned no metric scores",
                details={"backend": EvaluationBackend.DEEPEVAL.value},
            )
        return scores
