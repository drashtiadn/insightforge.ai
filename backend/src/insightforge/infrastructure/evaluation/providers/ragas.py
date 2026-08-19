"""RAGAS adapter (optional extra: ``uv sync --extra evaluation``)."""

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
from insightforge.shared.enums import EvaluationBackend

logger = get_logger(__name__)

# Injected scorer: sample in → metric-name → score. Used by unit tests.
RagasRunFn = Callable[[EvaluationSample], Mapping[str, float]]


def ragas_installed() -> bool:
    """True when the ``ragas`` package is importable (does not import it)."""

    return importlib.util.find_spec("ragas") is not None


class RagasEvaluator(Evaluator):
    """Score a sample with RAGAS, or an injected function in tests."""

    name = EvaluationBackend.RAGAS

    def __init__(
        self,
        *,
        run_fn: RagasRunFn | None = None,
        llm_configured: bool = False,
    ) -> None:
        self._run_fn = run_fn
        self._llm_configured = llm_configured

    @property
    def available(self) -> bool:
        if self._run_fn is not None:
            return True
        return ragas_installed() and self._llm_configured

    def evaluate(self, sample: EvaluationSample) -> EvaluationReport:
        if self._run_fn is None and not ragas_installed():
            raise ExternalServiceError(
                "ragas is not installed; install with: uv sync --extra evaluation",
                details={"backend": self.name.value},
            )
        if self._run_fn is None and not self._llm_configured:
            raise ExternalServiceError(
                "ragas requires OPENAI_API_KEY",
                details={"backend": self.name.value},
            )

        logger.info(
            "ragas evaluation started query=%r contexts=%d",
            sample.query,
            len(sample.contexts),
            extra={"backend": self.name.value, "contexts": len(sample.contexts)},
        )
        try:
            scores = dict(self._run_fn(sample) if self._run_fn is not None else _run_ragas(sample))
        except ExternalServiceError:
            raise
        except Exception as exc:
            logger.exception("ragas evaluation failed query=%r", sample.query)
            raise ExternalServiceError(
                f"ragas evaluation failed: {exc}",
                details={"backend": self.name.value},
            ) from exc

        logger.info(
            "ragas evaluation finished query=%r scores=%s",
            sample.query,
            scores,
            extra={"backend": self.name.value, "scores": dict(scores)},
        )
        return build_report(sample, backend=self.name, scores=scores)


def _run_ragas(sample: EvaluationSample) -> dict[str, float]:
    """Call RAGAS with a version-tolerant dataset/metrics mapping."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        dataset = _build_dataset(sample)
        metrics = _load_metrics()
        try:
            from ragas import evaluate
        except ImportError as exc:
            raise ExternalServiceError(
                "ragas is not installed; install with: uv sync --extra evaluation",
                details={"backend": EvaluationBackend.RAGAS.value},
            ) from exc

        try:
            result = evaluate(dataset, metrics=metrics)
        except TypeError:
            result = evaluate(dataset=dataset, metrics=metrics)
        return _scores_from_result(result)


def _build_dataset(sample: EvaluationSample) -> Any:
    dataset = _try_evaluation_dataset(sample)
    if dataset is not None:
        return dataset
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ExternalServiceError(
            "ragas dataset helpers are unavailable",
            details={"backend": EvaluationBackend.RAGAS.value},
        ) from exc
    return Dataset.from_dict(
        {
            "user_input": [sample.query],
            "question": [sample.query],
            "response": [sample.answer],
            "answer": [sample.answer],
            "retrieved_contexts": [list(sample.contexts)],
            "contexts": [list(sample.contexts)],
            "reference": [sample.reference],
            "ground_truth": [sample.reference],
        }
    )


def _try_evaluation_dataset(sample: EvaluationSample) -> Any | None:
    try:
        from ragas.dataset_schema import SingleTurnSample
    except ImportError:
        return None
    evaluation_dataset_cls: Any
    try:
        from ragas import EvaluationDataset as evaluation_dataset_cls
    except ImportError:
        return None
    try:
        return evaluation_dataset_cls(
            samples=[
                SingleTurnSample(
                    user_input=sample.query,
                    response=sample.answer,
                    retrieved_contexts=list(sample.contexts),
                    reference=sample.reference,
                )
            ]
        )
    except Exception:
        logger.debug("ragas EvaluationDataset construction failed", exc_info=True)
        return None


def _maybe_instantiate(metric: Any) -> Any:
    if isinstance(metric, type):
        return metric()
    return metric


def _load_metrics() -> list[Any]:
    try:
        from ragas.metrics import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )

        return [
            _maybe_instantiate(Faithfulness),
            _maybe_instantiate(AnswerRelevancy),
            _maybe_instantiate(ContextRecall),
            _maybe_instantiate(ContextPrecision),
        ]
    except ImportError:
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        return [
            _maybe_instantiate(faithfulness),
            _maybe_instantiate(answer_relevancy),
            _maybe_instantiate(context_recall),
            _maybe_instantiate(context_precision),
        ]


def _scores_from_result(result: Any) -> dict[str, float]:
    raw: Mapping[str, Any]
    scores = getattr(result, "scores", None)
    if isinstance(scores, dict):
        raw = scores
    elif isinstance(scores, list) and scores and isinstance(scores[0], dict):
        raw = scores[0]
    elif hasattr(result, "to_pandas"):
        frame = result.to_pandas()
        raw = frame.iloc[0].to_dict()
    elif isinstance(result, dict):
        raw = result
    else:
        try:
            raw = dict(result)
        except Exception as exc:
            raise ExternalServiceError(
                "ragas returned an unreadable result",
                details={"type": type(result).__name__},
            ) from exc

    parsed: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        parsed[str(key)] = float(value)
    if not parsed:
        raise ExternalServiceError(
            "ragas returned no numeric metric scores",
            details={"keys": list(raw)},
        )
    return parsed
