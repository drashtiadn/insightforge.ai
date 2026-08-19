"""Evaluator contract shared by every scoring backend."""

from __future__ import annotations

from abc import ABC, abstractmethod

from insightforge.domain.models import EvaluationReport, EvaluationSample
from insightforge.shared.enums import EvaluationBackend


class Evaluator(ABC):
    """One evaluation backend (RAGAS, DeepEval, or heuristic).

    Transport and library failures should raise ``ExternalServiceError`` so
    ``EvaluationService`` can fall back to the heuristic scorer.
    """

    name: EvaluationBackend

    @property
    def available(self) -> bool:
        """True when required packages/config are present and the backend can run."""

        return True

    @abstractmethod
    def evaluate(self, sample: EvaluationSample) -> EvaluationReport:
        """Score ``sample`` on faithfulness, relevancy, recall, and precision."""
