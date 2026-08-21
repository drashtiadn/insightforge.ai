"""LLM-as-judge quality gate (Phase 7.2).

After evaluation, the judge compares answer quality to a confidence
threshold. Failures can retry reasoning with a revision hint, up to a
bounded number of attempts. Gemini is used when available; otherwise a
deterministic heuristic compares evaluation scores to the threshold.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from insightforge.core.config import Settings, get_settings
from insightforge.core.exceptions import ExternalServiceError
from insightforge.core.logging import get_logger
from insightforge.domain.models import EvaluationReport, EvaluationSample, JudgeVerdict
from insightforge.infrastructure.evaluation.metrics import clamp_unit
from insightforge.infrastructure.llm import LlmMessage, LlmService, parse_model
from insightforge.prompts.evaluation import JUDGE_SYSTEM, JUDGE_USER

logger = get_logger(__name__)

_MAX_CONTEXT_CHARS = 400
_MAX_CONTEXTS = 4
_DEFAULT_HINT = (
    "Ground the answer more tightly in retrieved context and cover the query more completely."
)


class _LlmJudgePayload(BaseModel):
    passed: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    revision_hint: str = ""


def _quality_from_evaluation(
    evaluation: EvaluationReport | None,
    *,
    fallback: float,
) -> float:
    if evaluation is None:
        return clamp_unit(fallback)
    return clamp_unit(evaluation.overall)


def _issues_from_evaluation(
    evaluation: EvaluationReport | None,
    *,
    quality: float,
    threshold: float,
) -> list[str]:
    issues: list[str] = []
    if quality < threshold:
        issues.append(f"quality {quality:.2f} is below threshold {threshold:.2f}")
    if evaluation is None:
        return issues
    for metric in evaluation.metrics:
        if metric.score < threshold:
            issues.append(f"{metric.name.value} {metric.score:.2f} is below threshold")
    return issues


def _format_contexts(contexts: Sequence[str]) -> str:
    lines: list[str] = []
    for index, text in enumerate(contexts[:_MAX_CONTEXTS], start=1):
        snippet = text.strip().replace("\n", " ")
        if len(snippet) > _MAX_CONTEXT_CHARS:
            snippet = snippet[: _MAX_CONTEXT_CHARS - 1].rstrip() + "…"
        lines.append(f"{index}. {snippet}")
    return "\n".join(lines) or "(none)"


def _format_metrics(evaluation: EvaluationReport | None) -> str:
    if evaluation is None or not evaluation.metrics:
        return "(none)"
    return "\n".join(f"- {metric.name.value}: {metric.score:.2f}" for metric in evaluation.metrics)


class HeuristicJudge:
    """Compare evaluation / reflection scores to the confidence threshold."""

    name = "heuristic"

    def judge(
        self,
        sample: EvaluationSample,
        *,
        evaluation: EvaluationReport | None,
        reflection_confidence: float,
        threshold: float,
        attempt: int,
        max_retries: int,
    ) -> JudgeVerdict:
        quality = _quality_from_evaluation(evaluation, fallback=reflection_confidence)
        passed = quality >= threshold
        issues = _issues_from_evaluation(evaluation, quality=quality, threshold=threshold)
        retries_left = max(0, max_retries - attempt)
        retry = (not passed) and retries_left > 0
        hint = "" if passed else _DEFAULT_HINT
        logger.info(
            "heuristic judge passed=%s quality=%.3f threshold=%.3f retry=%s attempt=%d/%d",
            passed,
            quality,
            threshold,
            retry,
            attempt,
            max_retries,
            extra={
                "passed": passed,
                "quality": quality,
                "threshold": threshold,
                "retry": retry,
                "attempt": attempt,
            },
        )
        return JudgeVerdict(
            passed=passed,
            confidence=quality,
            threshold=threshold,
            quality=quality,
            issues=issues,
            revision_hint=hint,
            backend=self.name,
            retry=retry,
            attempt=attempt,
            max_retries=max_retries,
        )


class LlmJudge:
    """Gemini judge with heuristic fallback."""

    name = "llm"

    def __init__(self, llm: LlmService, *, fallback: HeuristicJudge | None = None) -> None:
        self._llm = llm
        self._fallback = fallback or HeuristicJudge()

    def judge(
        self,
        sample: EvaluationSample,
        *,
        evaluation: EvaluationReport | None,
        reflection_confidence: float,
        threshold: float,
        attempt: int,
        max_retries: int,
    ) -> JudgeVerdict:
        heuristic = self._fallback.judge(
            sample,
            evaluation=evaluation,
            reflection_confidence=reflection_confidence,
            threshold=threshold,
            attempt=attempt,
            max_retries=max_retries,
        )
        if not self._llm.available:
            logger.info("llm judge unavailable; using heuristic judge")
            return heuristic

        overall = evaluation.overall if evaluation is not None else reflection_confidence
        messages: list[LlmMessage] = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": JUDGE_USER.format(
                    query=sample.query,
                    answer=sample.answer or "(empty)",
                    contexts=_format_contexts(sample.contexts),
                    metrics=_format_metrics(evaluation),
                    overall=f"{overall:.2f}",
                    threshold=f"{threshold:.2f}",
                ),
            },
        ]
        try:
            raw = self._llm.complete(messages, json_mode=True)
            payload = parse_model(raw, _LlmJudgePayload)
        except ExternalServiceError as exc:
            logger.warning("llm judge failed; using heuristic judge error=%s", exc)
            return heuristic

        confidence = clamp_unit(payload.confidence)
        quality = _quality_from_evaluation(evaluation, fallback=confidence)
        passed = bool(payload.passed) and confidence >= threshold
        retries_left = max(0, max_retries - attempt)
        retry = (not passed) and retries_left > 0
        issues = [item.strip() for item in payload.issues if item.strip()] or heuristic.issues
        hint = payload.revision_hint.strip() or (heuristic.revision_hint if retry else "")
        logger.info(
            "llm judge passed=%s confidence=%.3f quality=%.3f threshold=%.3f retry=%s",
            passed,
            confidence,
            quality,
            threshold,
            retry,
            extra={
                "passed": passed,
                "confidence": confidence,
                "quality": quality,
                "retry": retry,
                "attempt": attempt,
            },
        )
        return JudgeVerdict(
            passed=passed,
            confidence=confidence,
            threshold=threshold,
            quality=quality,
            issues=issues,
            revision_hint=hint,
            backend=self.name,
            retry=retry,
            attempt=attempt,
            max_retries=max_retries,
        )


class JudgeService:
    """Facade used by the graph after evaluation."""

    def __init__(
        self,
        judge: HeuristicJudge | LlmJudge,
        *,
        enabled: bool = True,
        threshold: float = 0.5,
        max_retries: int = 1,
    ) -> None:
        self._judge = judge
        self._enabled = enabled
        self._threshold = clamp_unit(threshold)
        self._max_retries = max(0, max_retries)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def max_retries(self) -> int:
        return self._max_retries

    def judge(
        self,
        sample: EvaluationSample,
        *,
        evaluation: EvaluationReport | None = None,
        reflection_confidence: float = 0.0,
        attempt: int = 0,
    ) -> JudgeVerdict:
        """Return a pass/fail verdict and whether the graph should retry."""

        logger.info(
            "judge started backend=%s threshold=%.3f attempt=%d/%d query=%r",
            self._judge.name,
            self._threshold,
            attempt,
            self._max_retries,
            sample.query,
            extra={
                "backend": self._judge.name,
                "threshold": self._threshold,
                "attempt": attempt,
                "max_retries": self._max_retries,
            },
        )
        verdict = self._judge.judge(
            sample,
            evaluation=evaluation,
            reflection_confidence=reflection_confidence,
            threshold=self._threshold,
            attempt=attempt,
            max_retries=self._max_retries,
        )
        logger.info(
            "judge finished passed=%s retry=%s confidence=%.3f",
            verdict.passed,
            verdict.retry,
            verdict.confidence,
            extra={
                "passed": verdict.passed,
                "retry": verdict.retry,
                "confidence": verdict.confidence,
                "backend": verdict.backend,
            },
        )
        return verdict


def create_judge_service(
    settings: Settings | None = None,
    *,
    llm: LlmService | None = None,
    heuristic: bool = False,
) -> JudgeService:
    """Factory used by the graph and tests.

    ``heuristic=True`` forces the lexical judge (offline / stub runs).
    """

    cfg = settings or get_settings()
    if heuristic or llm is None or not llm.available:
        judge: HeuristicJudge | LlmJudge = HeuristicJudge()
    else:
        judge = LlmJudge(llm)
    logger.info(
        "judge service created backend=%s enabled=%s threshold=%.3f max_retries=%d",
        judge.name,
        cfg.judge_enabled,
        cfg.judge_confidence_threshold,
        cfg.judge_max_retries,
        extra={
            "backend": judge.name,
            "enabled": cfg.judge_enabled,
            "threshold": cfg.judge_confidence_threshold,
            "max_retries": cfg.judge_max_retries,
        },
    )
    return JudgeService(
        judge,
        enabled=cfg.judge_enabled,
        threshold=cfg.judge_confidence_threshold,
        max_retries=cfg.judge_max_retries,
    )
