"""Research pipeline endpoint."""

from fastapi import APIRouter

from insightforge.api.schemas import (
    ErrorResponse,
    EvaluationMetricScore,
    EvaluationResult,
    ResearchRequest,
    ResearchResponse,
    ResearchSource,
)
from insightforge.application.use_cases import ResearchRun, execute_research
from insightforge.domain.models import EvaluationReport

router = APIRouter(tags=["research"])


def _to_evaluation(report: EvaluationReport | None) -> EvaluationResult | None:
    if report is None:
        return None
    return EvaluationResult(
        backend=report.backend.value,
        overall=report.overall,
        metrics=[
            EvaluationMetricScore(
                name=metric.name.value,
                score=metric.score,
                reason=metric.reason,
            )
            for metric in report.metrics
        ],
        context_count=report.context_count,
        ground_truth_used=report.ground_truth_used,
    )


def _to_response(run: ResearchRun) -> ResearchResponse:
    return ResearchResponse(
        query=run.query,
        report=run.report,
        score=run.score,
        confidence=run.confidence,
        phase=run.phase,
        ok=run.ok,
        errors=list(run.errors),
        transitions=list(run.transitions),
        sources=[ResearchSource(title=source.title, url=source.url) for source in run.sources],
        evaluation=_to_evaluation(run.evaluation),
    )


@router.post(
    "/research",
    response_model=ResearchResponse,
    summary="Run a research query",
    responses={422: {"model": ErrorResponse, "description": "Invalid request body."}},
)
def run_research_endpoint(body: ResearchRequest) -> ResearchResponse:
    """Run the LangGraph research pipeline and return a cited markdown report.

    Live search and Gemini run when API keys are configured. This can take
    tens of seconds; set ``max_steps`` or ``stub_search`` for a shorter local try.
    Successful reports include automatic faithfulness, relevancy, recall, and
    precision scores when evaluation is enabled.
    """

    return _to_response(
        execute_research(
            body.query,
            max_steps=body.max_steps,
            stub_search=body.stub_search,
        )
    )
