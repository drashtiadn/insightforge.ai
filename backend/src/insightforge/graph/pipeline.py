"""Pipeline nodes — ingest, retrieve, reason, reflect, and report.

These run after the search loop and produce a cited ``ResearchReport``.
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
    try:
        result = agent.reason(state["query"], hits=hits, documents=documents)
    except Exception as exc:
        logger.warning("reasoner failed error=%s", exc)
        result = ReasoningResult(query=state["query"], answer="")

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
        **move(state, "done"),
    }
