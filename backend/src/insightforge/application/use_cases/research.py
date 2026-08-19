"""Run a research query through the LangGraph pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError

from insightforge.core.config import Settings, get_settings
from insightforge.core.exceptions import ValidationFailedError
from insightforge.domain.models import EvaluationReport
from insightforge.graph import WorkflowResult, run_research


@dataclass(frozen=True)
class ResearchSourceRef:
    """A cited source from a finished research run."""

    title: str
    url: str


@dataclass(frozen=True)
class ResearchRun:
    """HTTP-facing view of a finished research run."""

    query: str
    report: str
    score: float
    confidence: float
    phase: str
    ok: bool
    errors: tuple[str, ...]
    transitions: tuple[str, ...]
    sources: tuple[ResearchSourceRef, ...]
    evaluation: EvaluationReport | None = None


def _sources_from_result(result: WorkflowResult) -> tuple[ResearchSourceRef, ...]:
    seen: set[str] = set()
    sources: list[ResearchSourceRef] = []
    for raw in result.state["sources"]:
        url = str(raw.get("url") or "")
        title = str(raw.get("title") or "")
        if not url or url in seen:
            continue
        seen.add(url)
        sources.append(ResearchSourceRef(title=title, url=url))
    return tuple(sources)


def _unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def _evaluation_from_result(result: WorkflowResult) -> EvaluationReport | None:
    raw = result.state.get("evaluation") or {}
    if not raw or "metrics" not in raw:
        return None
    try:
        return EvaluationReport.model_validate(raw)
    except ValidationError:
        return None


def _to_run(result: WorkflowResult) -> ResearchRun:
    return ResearchRun(
        query=result.query,
        report=result.report,
        score=_unit(result.score),
        confidence=_unit(result.confidence),
        phase=result.phase,
        ok=result.ok,
        errors=result.errors,
        transitions=result.transitions,
        sources=_sources_from_result(result),
        evaluation=_evaluation_from_result(result),
    )


def execute_research(
    query: str,
    *,
    max_steps: int | None = None,
    stub_search: bool = False,
    settings: Settings | None = None,
) -> ResearchRun:
    """Execute the research pipeline and map the result for the API.

    ``stub_search`` uses offline example sources. It is ignored in production
    so clients cannot skip live retrieval.
    """

    resolved = settings or get_settings()
    use_stub = stub_search and not resolved.is_production
    result = run_research(query, max_steps=max_steps, stub_search=use_stub)
    if result.phase == "failed" and any("must not be empty" in error for error in result.errors):
        raise ValidationFailedError(
            "Query must not be empty",
            details={"field": "query"},
        )
    return _to_run(result)
