"""RAG report evaluation — RAGAS, DeepEval, heuristic scores, and LLM judge."""

from insightforge.infrastructure.evaluation.base import Evaluator
from insightforge.infrastructure.evaluation.judge import (
    HeuristicJudge,
    JudgeService,
    LlmJudge,
    create_judge_service,
)
from insightforge.infrastructure.evaluation.metrics import (
    append_evaluation_section,
    build_report,
    clamp_unit,
    insert_section_before_errors,
)
from insightforge.infrastructure.evaluation.providers import (
    DeepEvalEvaluator,
    DeepEvalRunFn,
    HeuristicEvaluator,
    RagasEvaluator,
    RagasRunFn,
    deepeval_installed,
    ragas_installed,
)
from insightforge.infrastructure.evaluation.sample import (
    build_sample,
    contexts_from_hits_and_documents,
)
from insightforge.infrastructure.evaluation.service import (
    EvaluationService,
    build_provider,
    create_evaluation_service,
    openai_configured,
    parse_evaluation_backend,
    resolve_backend,
)

__all__ = [
    "DeepEvalEvaluator",
    "DeepEvalRunFn",
    "EvaluationService",
    "Evaluator",
    "HeuristicEvaluator",
    "HeuristicJudge",
    "JudgeService",
    "LlmJudge",
    "RagasEvaluator",
    "RagasRunFn",
    "append_evaluation_section",
    "build_provider",
    "build_report",
    "build_sample",
    "clamp_unit",
    "contexts_from_hits_and_documents",
    "create_evaluation_service",
    "create_judge_service",
    "deepeval_installed",
    "insert_section_before_errors",
    "openai_configured",
    "parse_evaluation_backend",
    "ragas_installed",
    "resolve_backend",
]
