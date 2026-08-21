"""Pipeline nodes — ingest, retrieve, reason, reflect, report, and evaluate.

These run after the search loop and produce a cited, quality-scored report.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from insightforge.agents.planner.schemas import QueryAnalysis, ResearchPlan
from insightforge.agents.reasoner import Reasoner, SimpleReasoner
from insightforge.agents.reflection import ReflectionAgent, SimpleReflectionAgent
from insightforge.agents.report import ReportGenerator, SimpleReportGenerator
from insightforge.core.logging import get_logger
from insightforge.domain.models import (
    Document,
    DocumentChunk,
    EvaluationReport,
    ParsedDocument,
    ReasoningResult,
    ReflectionResult,
    ResearchTask,
    RetrievalHit,
)
from insightforge.graph.helpers import move
from insightforge.graph.state import GraphState
from insightforge.infrastructure.document.chunking import ChunkConfig, chunk_document
from insightforge.infrastructure.document.citation import cite_chunks, cite_document
from insightforge.infrastructure.evaluation import (
    EvaluationService,
    JudgeService,
    append_evaluation_section,
    build_sample,
    contexts_from_hits_and_documents,
    create_evaluation_service,
    create_judge_service,
    insert_section_before_errors,
)
from insightforge.infrastructure.rerankers import RerankerService
from insightforge.infrastructure.retrieval import RetrievalService
from insightforge.shared.enums import ContentType, QueryIntent, RetrievalMode, SearchProviderHint

logger = get_logger(__name__)

_MAX_FOLLOW_UPS = 1


def _documents_from_state(state: GraphState) -> list[Document]:
    """Prefer search documents; synthesize from stub sources when needed."""

    documents: list[Document] = []
    for raw in state["documents"]:
        try:
            documents.append(Document.model_validate(raw))
        except ValidationError as exc:
            logger.warning("skipping invalid document error=%s", exc)
    if documents:
        return documents

    synthesized: list[Document] = []
    for index, source in enumerate(state["sources"], start=1):
        title = str(source.get("title") or f"Source {index}")
        url = str(source.get("url") or f"https://example.com/research/{index}")
        text = f"{title}. {state['query']}"
        synthesized.append(
            Document(
                title=title,
                url=url,
                snippet=text,
                content=text,
                provider=SearchProviderHint.WEB,
            )
        )
    return synthesized


def _plan_from_state(state: GraphState) -> ResearchPlan:
    try:
        intent = QueryIntent(state["intent"]) if state["intent"] else QueryIntent.EXPLORATORY
    except ValueError:
        intent = QueryIntent.EXPLORATORY
    tasks: list[ResearchTask] = []
    for raw in state["tasks"]:
        try:
            tasks.append(ResearchTask.model_validate(raw))
        except ValidationError:
            continue
    analysis = QueryAnalysis(
        original_query=state["query"],
        normalized_query=state["query"],
        keywords=[],
        token_count=0,
    )
    return ResearchPlan(
        query=state["query"],
        analysis=analysis,
        intent=intent,
        tasks=tasks,
    )


def ingest_node(
    state: GraphState,
    *,
    retrieval: RetrievalService | None = None,
    chunk_config: ChunkConfig | None = None,
) -> dict[str, Any]:
    """Parse/chunk gathered documents and index them for retrieval."""

    documents = _documents_from_state(state)
    chunks: list[DocumentChunk] = []
    for document in documents:
        text = (document.content or document.snippet or document.title).strip()
        if not text:
            continue
        parsed = ParsedDocument(
            text=text,
            content_type=ContentType.MARKDOWN,
            title=document.title,
            url=document.url,
            metadata={"provider": document.provider.value},
        )
        parsed = cite_document(parsed)
        produced = chunk_document(parsed, config=chunk_config)
        chunks.extend(cite_chunks(produced, parsed))

    logger.info("ingest node chunks=%d documents=%d", len(chunks), len(documents))
    if retrieval is not None:
        retrieval.clear()
        if chunks:
            texts = [chunk.text for chunk in chunks]
            ids = [f"c{index}" for index in range(len(chunks))]
            metadata: list[dict[str, Any]] = [
                {
                    "title": chunk.title or "",
                    "url": chunk.url or "",
                    "heading": chunk.heading or "",
                }
                for chunk in chunks
            ]
            embedder = retrieval.embeddings
            if embedder is not None and embedder.available:
                retrieval.index_texts(texts, ids=ids, metadata=metadata)
            else:
                retrieval.index_lexical(texts, ids=ids, metadata=metadata)

    return {
        "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
        **move(state, "ingest"),
    }


def retrieve_node(
    state: GraphState,
    *,
    retrieval: RetrievalService | None = None,
    reranker: RerankerService | None = None,
) -> dict[str, Any]:
    """Retrieve (and optionally rerank) chunks for the query."""

    hits: list[RetrievalHit] = []
    if retrieval is not None:
        embedder = retrieval.embeddings
        mode = (
            RetrievalMode.HYBRID
            if embedder is not None and embedder.available
            else RetrievalMode.BM25
        )
        try:
            hits = retrieval.retrieve(state["query"], mode=mode)
        except Exception as exc:
            logger.warning("retrieve failed error=%s", exc)
            hits = []

        if reranker is not None and reranker.available and hits:
            try:
                hits = reranker.rerank(state["query"], hits)
            except Exception as exc:
                logger.warning("rerank skipped error=%s", exc)

    if not hits and state["chunks"]:
        for index, raw in enumerate(state["chunks"], start=1):
            text = str(raw.get("text") or "")
            if not text:
                continue
            hits.append(
                RetrievalHit(
                    id=f"c{index - 1}",
                    score=1.0 / index,
                    text=text,
                    metadata={
                        "title": raw.get("title") or "",
                        "url": raw.get("url") or "",
                    },
                    mode=RetrievalMode.BM25,
                    bm25_rank=index,
                )
            )

    logger.info("retrieve node hits=%d", len(hits))
    return {
        "hits": [hit.model_dump(mode="json") for hit in hits],
        **move(state, "retrieve"),
    }


def reason_node(
    state: GraphState,
    *,
    reasoner: Reasoner | None = None,
) -> dict[str, Any]:
    """Synthesize an answer from retrieval hits and search documents."""

    agent = reasoner or SimpleReasoner()
    hits = [RetrievalHit.model_validate(raw) for raw in state["hits"]]
    documents = _documents_from_state(state)
    feedback = state["revision_hint"].strip() or None
    try:
        result = agent.reason(state["query"], hits=hits, documents=documents, feedback=feedback)
    except Exception as exc:
        logger.warning("reasoner failed error=%s", exc)
        result = ReasoningResult(query=state["query"], answer="")

    if feedback:
        logger.info("reason node applied revision hint chars=%d", len(feedback))

    logger.info(
        "reason node confidence=%.2f clusters=%d",
        result.confidence,
        len(result.clusters),
    )
    return {
        "reasoning": result.model_dump(mode="json"),
        "score": result.confidence,
        **move(state, "reason"),
    }


def reflect_node(
    state: GraphState,
    *,
    reflection: ReflectionAgent | None = None,
) -> dict[str, Any]:
    """Score sufficiency and optionally schedule one follow-up search task."""

    agent = reflection or SimpleReflectionAgent()
    try:
        reasoning = ReasoningResult.model_validate(state["reasoning"])
    except ValidationError:
        reasoning = ReasoningResult(query=state["query"], answer="")

    hits = [RetrievalHit.model_validate(raw) for raw in state["hits"]]
    documents = _documents_from_state(state)
    verdict = agent.reflect(reasoning, hits=hits, documents=documents)

    updates: dict[str, Any] = {
        "reflection": verdict.model_dump(mode="json"),
        "score": verdict.confidence,
        **move(state, "reflect"),
    }

    if (
        not verdict.is_sufficient
        and verdict.follow_up_queries
        and state["follow_up_used"] < _MAX_FOLLOW_UPS
    ):
        follow_query = verdict.follow_up_queries[0]
        task = ResearchTask(
            id=f"follow-{state['follow_up_used'] + 1}",
            description=f"Follow-up: {follow_query}",
            search_query=follow_query,
            providers=[SearchProviderHint.WEB, SearchProviderHint.WIKIPEDIA],
            priority=9,
        )
        tasks = [*state["tasks"], task.model_dump(mode="json")]
        updates["tasks"] = tasks
        updates["max_steps"] = max(state["max_steps"], len(tasks))
        updates["follow_up_used"] = state["follow_up_used"] + 1
        logger.info("reflection scheduled follow-up query=%r", follow_query)

    return updates


def report_node(
    state: GraphState,
    *,
    reporter: ReportGenerator | None = None,
) -> dict[str, Any]:
    """Build a Markdown research report from reasoning and reflection."""

    agent = reporter or SimpleReportGenerator()
    plan = _plan_from_state(state)
    try:
        reasoning = ReasoningResult.model_validate(state["reasoning"])
    except ValidationError:
        reasoning = ReasoningResult(query=state["query"], answer="")
    reflection: ReflectionResult | None = None
    if state["reflection"]:
        try:
            reflection = ReflectionResult.model_validate(state["reflection"])
        except ValidationError:
            reflection = None

    hits = [RetrievalHit.model_validate(raw) for raw in state["hits"]]
    documents = _documents_from_state(state)
    report = agent.generate(
        plan=plan,
        reasoning=reasoning,
        reflection=reflection,
        hits=hits,
        documents=documents,
    )
    markdown = report.to_markdown()
    if state["errors"]:
        error_lines = "\n".join(f"- {err}" for err in state["errors"])
        markdown += f"\n## Errors\n\n{error_lines}\n"

    logger.info(
        "report node sections=%d citations=%d confidence=%.2f",
        len(report.sections),
        len(report.citations),
        report.confidence,
    )
    return {
        "report": markdown,
        "score": report.confidence if report.confidence else state["score"],
        **move(state, "report"),
    }


def evaluate_node(
    state: GraphState,
    *,
    evaluator: EvaluationService | None = None,
) -> dict[str, Any]:
    """Score the generated answer against retrieved context (Phase 7.1)."""

    service = evaluator or create_evaluation_service()
    if not service.enabled:
        logger.info("evaluation skipped (disabled)")
        return {"evaluation": {}, **move(state, "evaluate")}

    try:
        reasoning = (
            ReasoningResult.model_validate(state["reasoning"]) if state["reasoning"] else None
        )
    except ValidationError:
        reasoning = None
    answer = reasoning.answer.strip() if reasoning is not None else ""
    if not answer:
        answer = state["report"]

    hits = [RetrievalHit.model_validate(raw) for raw in state["hits"]]
    documents = _documents_from_state(state)
    sample = build_sample(
        query=state["query"],
        answer=answer,
        contexts=contexts_from_hits_and_documents(hits, documents),
    )
    try:
        report = service.evaluate(sample)
    except Exception as exc:
        logger.warning("evaluation failed error=%s", exc)
        return {
            "evaluation": {"error": str(exc)},
            **move(state, "evaluate"),
        }

    markdown = state["report"]
    if service.append_to_report and markdown:
        markdown = append_evaluation_section(markdown, report)

    logger.info(
        "evaluate node backend=%s overall=%.2f contexts=%d",
        report.backend.value,
        report.overall,
        report.context_count,
        extra={
            "backend": report.backend.value,
            "overall": report.overall,
            "contexts": report.context_count,
        },
    )
    return {
        "evaluation": report.model_dump(mode="json"),
        "report": markdown,
        **move(state, "evaluate"),
    }


def _evaluation_from_state(state: GraphState) -> EvaluationReport | None:
    raw = state["evaluation"] or {}
    if not raw or "metrics" not in raw:
        return None
    try:
        return EvaluationReport.model_validate(raw)
    except ValidationError:
        return None


def _reflection_confidence(state: GraphState) -> float:
    raw = state["reflection"].get("confidence") if state["reflection"] else None
    if isinstance(raw, int | float):
        return float(raw)
    return float(state["score"] or 0.0)


def judge_node(
    state: GraphState,
    *,
    judge: JudgeService | None = None,
) -> dict[str, Any]:
    """LLM-as-judge confidence gate; retry reasoning when below threshold."""

    service = judge or create_judge_service(heuristic=True)
    if not service.enabled:
        logger.info("judge skipped (disabled)")
        return {"judgment": {}, **move(state, "done")}

    try:
        reasoning = (
            ReasoningResult.model_validate(state["reasoning"]) if state["reasoning"] else None
        )
    except ValidationError:
        reasoning = None
    answer = reasoning.answer.strip() if reasoning is not None else ""
    if not answer:
        answer = state["report"]

    hits = [RetrievalHit.model_validate(raw) for raw in state["hits"]]
    documents = _documents_from_state(state)
    sample = build_sample(
        query=state["query"],
        answer=answer,
        contexts=contexts_from_hits_and_documents(hits, documents),
    )
    attempt = state["judge_retries"]
    try:
        verdict = service.judge(
            sample,
            evaluation=_evaluation_from_state(state),
            reflection_confidence=_reflection_confidence(state),
            attempt=attempt,
        )
    except Exception as exc:
        logger.warning("judge failed error=%s", exc)
        return {
            "judgment": {"error": str(exc), "passed": False, "retry": False},
            **move(state, "done"),
        }

    if verdict.retry:
        next_attempt = attempt + 1
        logger.info(
            "judge retrying reason attempt=%d/%d hint=%r",
            next_attempt,
            verdict.max_retries,
            verdict.revision_hint,
        )
        return {
            "judgment": verdict.model_dump(mode="json"),
            "revision_hint": verdict.revision_hint,
            "judge_retries": next_attempt,
            **move(state, "judge"),
        }

    markdown = state["report"]
    if markdown:
        markdown = insert_section_before_errors(markdown, verdict.to_markdown())
    logger.info(
        "judge node passed=%s confidence=%.2f backend=%s",
        verdict.passed,
        verdict.confidence,
        verdict.backend,
        extra={
            "passed": verdict.passed,
            "confidence": verdict.confidence,
            "backend": verdict.backend,
            "attempt": verdict.attempt,
        },
    )
    return {
        "judgment": verdict.model_dump(mode="json"),
        "report": markdown,
        **move(state, "done"),
    }
