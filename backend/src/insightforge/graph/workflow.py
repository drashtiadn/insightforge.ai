"""Build, compile, and run the research workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast
from uuid import uuid4

from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langsmith import traceable

from insightforge.agents import (
    Planner,
    Reasoner,
    ReflectionAgent,
    ReportGenerator,
    create_planner,
    create_reasoner,
    create_reflection_agent,
    create_report_generator,
)
from insightforge.core.config import get_settings
from insightforge.core.logging import get_logger
from insightforge.graph.edges import after_judge, after_plan, after_reflect, after_search
from insightforge.graph.nodes import plan_node, research_node, search_node
from insightforge.graph.pipeline import (
    evaluate_node,
    ingest_node,
    judge_node,
    reason_node,
    reflect_node,
    report_node,
    retrieve_node,
)
from insightforge.graph.state import GraphState, initial_state
from insightforge.infrastructure.document.chunking import ChunkConfig
from insightforge.infrastructure.document.service import chunk_config_from_settings
from insightforge.infrastructure.evaluation import (
    EvaluationService,
    HeuristicEvaluator,
    JudgeService,
    create_evaluation_service,
    create_judge_service,
)
from insightforge.infrastructure.llm import LlmService, create_llm_service
from insightforge.infrastructure.rerankers import RerankerService, create_reranker_service
from insightforge.infrastructure.retrieval import RetrievalService, create_retrieval_service
from insightforge.infrastructure.search import SearchService, create_search_service
from insightforge.infrastructure.tracing import (
    configure_tracing,
    debug_run_summary,
    tracing_status,
)
from insightforge.infrastructure.vectorstores.stores import MemoryVectorStore

logger = get_logger(__name__)


@dataclass(frozen=True)
class WorkflowResult:
    """Simple view of a finished research run."""

    query: str
    report: str
    score: float
    confidence: float
    phase: str
    errors: tuple[str, ...]
    transitions: tuple[str, ...]
    state: GraphState

    @property
    def ok(self) -> bool:
        """True when the run finished a report without recorded errors."""

        return self.phase == "done" and not self.errors


@dataclass
class PipelineResources:
    """Services bound into a compiled graph for one run or a long-lived graph."""

    search: SearchService | None
    owns_search: bool
    retrieval_root: RetrievalService
    session: RetrievalService
    reranker: RerankerService | None
    evaluator: EvaluationService
    judge: JudgeService
    llm: LlmService
    planner: Planner
    reasoner: Reasoner
    reporter: ReportGenerator
    reflection: ReflectionAgent
    chunk_config: ChunkConfig

    def close(self) -> None:
        """Close owned HTTP clients. Idempotent."""

        if self.owns_search and self.search is not None:
            self.search.close()
        self.llm.close()
        if self.reranker is not None:
            self.reranker.close()
        embeddings = self.retrieval_root.embeddings
        if embeddings is not None:
            embeddings.close()


def resolve_search_service(
    search_service: SearchService | None = None,
    *,
    stub_search: bool = False,
) -> tuple[SearchService | None, bool]:
    """Resolve the search service used by the default graph.

    Returns ``(service, owns_service)``. When ``stub_search`` is True, the
    offline stub path is used (``service`` is ``None``). Otherwise an omitted
    service is created via ``create_search_service()`` and marked owned.
    """

    if stub_search:
        return None, False
    if search_service is not None:
        return search_service, False
    return create_search_service(), True


def assemble_resources(
    *,
    search_service: SearchService | None = None,
    stub_search: bool = False,
    retrieval: RetrievalService | None = None,
    reranker: RerankerService | None = None,
    evaluator: EvaluationService | None = None,
    judge: JudgeService | None = None,
    llm: LlmService | None = None,
    planner: Planner | None = None,
    reasoner: Reasoner | None = None,
    reporter: ReportGenerator | None = None,
    reflection: ReflectionAgent | None = None,
) -> PipelineResources:
    """Build default pipeline services for compile/run."""

    search, owns_search = resolve_search_service(search_service, stub_search=stub_search)
    llm_service = llm if llm is not None else create_llm_service()
    if retrieval is not None:
        retrieval_root = retrieval
        session = retrieval_root
    elif stub_search:
        retrieval_root = RetrievalService(MemoryVectorStore())
        session = retrieval_root
    else:
        retrieval_root = create_retrieval_service()
        session = retrieval_root.session(uuid4().hex)
    rerank_service = reranker
    if rerank_service is None and not stub_search:
        rerank_service = create_reranker_service()
    if evaluator is not None:
        evaluation_service = evaluator
    elif stub_search:
        cfg = get_settings()
        evaluation_service = EvaluationService(
            HeuristicEvaluator(),
            enabled=cfg.evaluation_enabled,
            append_to_report=cfg.evaluation_append_to_report,
        )
        logger.info(
            "evaluation using heuristic backend for stub_search enabled=%s",
            cfg.evaluation_enabled,
        )
    else:
        evaluation_service = create_evaluation_service()
    if judge is not None:
        judge_service = judge
    else:
        judge_service = create_judge_service(llm=llm_service, heuristic=stub_search)
    return PipelineResources(
        search=search,
        owns_search=owns_search,
        retrieval_root=retrieval_root,
        session=session,
        reranker=rerank_service,
        evaluator=evaluation_service,
        judge=judge_service,
        llm=llm_service,
        planner=planner or create_planner(llm_service),
        reasoner=reasoner or create_reasoner(llm_service),
        reporter=reporter or create_report_generator(llm_service),
        reflection=reflection or create_reflection_agent(llm_service),
        chunk_config=chunk_config_from_settings(get_settings()),
    )


def build_graph(
    *,
    search_service: SearchService | None = None,
    resources: PipelineResources | None = None,
) -> StateGraph[GraphState, None, GraphState, GraphState]:
    """Wire nodes and edges into a StateGraph.

    ``resources`` binds planner, retrieval, reasoner, reflection, and report.
    When omitted, heuristic defaults run (offline / tests).

    Flow::

        START → plan → research → search ⇄ ingest → retrieve → reason → reflect ⇄ report → evaluate → judge ⇄ reason → END
    """

    deps = resources
    search = deps.search if deps is not None else search_service

    # LangGraph's add_node overloads do not accept a plain Callable annotation.
    plan: Any
    ingest: Any
    retrieve: Any
    reason: Any
    reflect: Any
    report: Any
    evaluate: Any
    judge_step: Any

    if deps is None:
        plan = plan_node
        ingest = ingest_node
        retrieve = retrieve_node
        reason = reason_node
        reflect = reflect_node
        report = report_node
        evaluate = evaluate_node
        judge_step = judge_node
    else:

        def plan(state: GraphState) -> dict[str, Any]:
            return plan_node(state, planner=deps.planner)

        def ingest(state: GraphState) -> dict[str, Any]:
            return ingest_node(
                state,
                retrieval=deps.session,
                chunk_config=deps.chunk_config,
            )

        def retrieve(state: GraphState) -> dict[str, Any]:
            return retrieve_node(
                state,
                retrieval=deps.session,
                reranker=deps.reranker,
            )

        def reason(state: GraphState) -> dict[str, Any]:
            return reason_node(state, reasoner=deps.reasoner)

        def reflect(state: GraphState) -> dict[str, Any]:
            return reflect_node(state, reflection=deps.reflection)

        def report(state: GraphState) -> dict[str, Any]:
            return report_node(state, reporter=deps.reporter)

        def evaluate(state: GraphState) -> dict[str, Any]:
            return evaluate_node(state, evaluator=deps.evaluator)

        def judge_step(state: GraphState) -> dict[str, Any]:
            return judge_node(state, judge=deps.judge)

    graph: StateGraph[GraphState, None, GraphState, GraphState] = StateGraph(GraphState)
    graph.add_node("plan", plan)
    graph.add_node("research", research_node)
    graph.add_node("search", lambda state: search_node(state, search_service=search))
    graph.add_node("ingest", ingest)
    graph.add_node("retrieve", retrieve)
    graph.add_node("reason", reason)
    graph.add_node("reflect", reflect)
    graph.add_node("report", report)
    graph.add_node("evaluate", evaluate)
    graph.add_node("judge", judge_step)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", after_plan)
    graph.add_edge("research", "search")
    graph.add_conditional_edges("search", after_search)
    graph.add_edge("ingest", "retrieve")
    graph.add_edge("retrieve", "reason")
    graph.add_edge("reason", "reflect")
    graph.add_conditional_edges("reflect", after_reflect)
    graph.add_edge("report", "evaluate")
    graph.add_edge("evaluate", "judge")
    graph.add_conditional_edges("judge", after_judge)
    return graph


def compile_graph(
    *,
    search_service: SearchService | None = None,
    stub_search: bool = False,
    resources: PipelineResources | None = None,
) -> CompiledStateGraph[GraphState, None, GraphState, GraphState]:
    """Return a runnable graph with pipeline services injected by default."""

    deps = resources or assemble_resources(
        search_service=search_service,
        stub_search=stub_search,
    )
    return build_graph(search_service=deps.search, resources=deps).compile()


@traceable(name="run_research", run_type="chain", tags=["insightforge", "research"])
def run_research(
    query: str,
    *,
    max_steps: int | None = None,
    max_retries: int = 3,
    search_service: SearchService | None = None,
    stub_search: bool = False,
) -> WorkflowResult:
    """Run the full research workflow and return a structured result.

    When LangSmith tracing is enabled, this run and the LangGraph node spans
    appear in the configured project for debugging.

    ``max_steps=None`` runs every planned research task (resolved after plan).
    A ``SearchService`` is injected by default; pass ``stub_search=True`` to
    force the offline stub fetcher. Gemini is used when ``GEMINI_API_KEY`` is set.
    """

    configure_tracing()
    resources = assemble_resources(
        search_service=search_service,
        stub_search=stub_search,
    )
    try:
        state = cast(
            GraphState,
            build_graph(search_service=resources.search, resources=resources)
            .compile()
            .invoke(initial_state(query, max_steps=max_steps, max_retries=max_retries)),
        )
    finally:
        resources.close()

    confidence = state["score"]
    if state["reflection"]:
        raw = state["reflection"].get("confidence")
        if isinstance(raw, int | float):
            confidence = float(raw)
    if state["judgment"]:
        raw = state["judgment"].get("confidence")
        if isinstance(raw, int | float):
            confidence = float(raw)

    result = WorkflowResult(
        query=state["query"],
        report=state["report"],
        score=state["score"],
        confidence=confidence,
        phase=state["phase"],
        errors=tuple(state["errors"]),
        transitions=tuple(state["transitions"]),
        state=state,
    )

    summary = debug_run_summary(
        phase=result.phase,
        ok=result.ok,
        score=result.score,
        errors=result.errors,
        transitions=result.transitions,
    )
    evaluation = state.get("evaluation") or {}
    overall = evaluation.get("overall")
    if isinstance(overall, int | float):
        summary["evaluation_overall"] = float(overall)
        summary["evaluation_backend"] = evaluation.get("backend")
    judgment = state.get("judgment") or {}
    if "passed" in judgment:
        summary["judge_passed"] = judgment.get("passed")
        summary["judge_confidence"] = judgment.get("confidence")
    if tracing_status().enabled:
        logger.info(
            "research run traced query=%r phase=%s ok=%s score=%s",
            result.query,
            result.phase,
            result.ok,
            result.score,
            extra=summary,
        )
    else:
        logger.debug(
            "research run finished query=%r phase=%s ok=%s",
            result.query,
            result.phase,
            result.ok,
            extra=summary,
        )

    return result
