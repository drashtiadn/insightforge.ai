"""Deterministic report generator used for offline runs and CI.

Deterministic sectioning driven by the reasoner output and evidence. The
Phase 6.2 citation generator will replace the naive citation assembly below;
sections and the public shape stay the same.
"""

from __future__ import annotations

from collections.abc import Sequence

from insightforge.agents.planner.schemas import ResearchPlan
from insightforge.agents.report.base import ReportGenerator
from insightforge.core.logging import get_logger
from insightforge.domain.models import (
    Citation,
    Document,
    ReasoningResult,
    ReflectionResult,
    ReportSection,
    ResearchReport,
    RetrievalHit,
)

logger = get_logger(__name__)


def _title_from_query(query: str) -> str:
    cleaned = " ".join(query.strip().split())
    if not cleaned:
        return "Research Report"
    return cleaned[:1].upper() + cleaned[1:]


def _citations_from_documents(documents: Sequence[Document]) -> list[Citation]:
    citations: list[Citation] = []
    seen_urls: set[str] = set()
    for doc in documents:
        url = doc.url.strip() if doc.url else ""
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        citations.append(
            Citation(
                url=url,
                title=doc.title or None,
                site_name=doc.provider.value,
            )
        )
    return citations


def _citations_from_hits(hits: Sequence[RetrievalHit]) -> list[Citation]:
    citations: list[Citation] = []
    for hit in hits:
        title = str(hit.metadata.get("title") or hit.id)
        url = hit.metadata.get("url")
        citations.append(
            Citation(
                url=str(url) if url else None,
                title=title,
                locator=f"chunk {hit.id}",
            )
        )
    return citations


class SimpleReportGenerator(ReportGenerator):
    """Assemble a Markdown-ready report from plan, reasoning, and evidence."""

    def generate(
        self,
        *,
        plan: ResearchPlan,
        reasoning: ReasoningResult,
        reflection: ReflectionResult | None = None,
        hits: Sequence[RetrievalHit] | None = None,
        documents: Sequence[Document] | None = None,
    ) -> ResearchReport:
        hit_list = list(hits or [])
        doc_list = list(documents or [])
        logger.info(
            "report generation started intent=%s tasks=%d hits=%d docs=%d",
            plan.intent.value,
            len(plan.tasks),
            len(hit_list),
            len(doc_list),
        )

        citations: list[Citation] = []
        citations.extend(_citations_from_documents(doc_list))
        citations.extend(_citations_from_hits(hit_list))

        sections: list[ReportSection] = []

        plan_lines = [f"- {step}" for step in plan.steps] or ["- (no plan steps recorded)"]
        sections.append(
            ReportSection(
                heading="Research Plan",
                body="\n".join(plan_lines),
            )
        )

        summary_body = reasoning.answer.strip() or (
            "No supporting evidence was gathered for this query."
        )
        sections.append(
            ReportSection(
                heading="Summary",
                body=summary_body,
                citation_indexes=list(range(1, len(citations) + 1)),
            )
        )

        if reasoning.key_points:
            key_point_body = "\n".join(f"- {point}" for point in reasoning.key_points)
            sections.append(
                ReportSection(
                    heading="Key Findings",
                    body=key_point_body,
                )
            )

        if reflection is not None:
            reflection_lines = [
                f"Sufficient: {'yes' if reflection.is_sufficient else 'no'}",
                f"Confidence: {reflection.confidence:.2f}",
            ]
            if reflection.gaps:
                reflection_lines.append("")
                reflection_lines.append("Gaps:")
                reflection_lines.extend(f"- {gap}" for gap in reflection.gaps)
            if reflection.follow_up_queries:
                reflection_lines.append("")
                reflection_lines.append("Suggested follow-ups:")
                reflection_lines.extend(f"- {query}" for query in reflection.follow_up_queries)
            sections.append(
                ReportSection(
                    heading="Reflection",
                    body="\n".join(reflection_lines),
                )
            )

        confidence = reflection.confidence if reflection is not None else reasoning.confidence
        title = _title_from_query(plan.query or reasoning.query)
        summary = summary_body if summary_body != reasoning.answer.strip() else reasoning.answer

        report = ResearchReport(
            query=plan.query or reasoning.query,
            title=title,
            summary=summary,
            sections=sections,
            citations=citations,
            confidence=confidence,
            metadata={
                "intent": plan.intent.value,
                "task_count": len(plan.tasks),
                "hit_count": len(hit_list),
                "document_count": len(doc_list),
                "reflection_sufficient": (None if reflection is None else reflection.is_sufficient),
            },
        )

        logger.info(
            "report generation complete sections=%d citations=%d confidence=%.2f",
            len(sections),
            len(citations),
            confidence,
            extra={
                "sections": len(sections),
                "citations": len(citations),
                "confidence": confidence,
            },
        )
        return report
