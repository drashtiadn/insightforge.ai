"""Deterministic report generator used for offline runs and CI.

Consumes ``ReasoningResult`` (including its clusters and conflicts) plus an
optional ``ReflectionResult`` and builds a Markdown-ready ``ResearchReport``:

- Summary section carries the synthesized answer.
- Key Findings section prints each claim with its citation indexes.
- Evidence section lists supporting snippets under each claim.
- Conflicts and Reflection sections appear when data is available.

Citations are generated once via ``research.citations.build_citations`` and
each finding is linked to the citation indexes that back it — no free-form
citation strings scattered across sections.
"""

from __future__ import annotations

from collections.abc import Sequence

from insightforge.agents.planner.schemas import ResearchPlan
from insightforge.agents.report.base import ReportGenerator
from insightforge.core.logging import get_logger
from insightforge.domain.models import (
    Document,
    EvidenceCluster,
    ReasoningResult,
    ReflectionResult,
    ReportSection,
    ResearchReport,
    RetrievalHit,
)
from insightforge.research import (
    aggregate_evidence,
    build_citations,
    link_clusters_to_citations,
)

logger = get_logger(__name__)


def _title_from_query(query: str) -> str:
    cleaned = " ".join(query.strip().split())
    if not cleaned:
        return "Research Report"
    return cleaned[:1].upper() + cleaned[1:]


class SimpleReportGenerator(ReportGenerator):
    """Assemble a Markdown-ready report backed by the research feature toolkit."""

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

        # Prefer the reasoner's clusters (they already reflect compression and
        # aggregation). Fall back to aggregating from raw evidence so this
        # agent still works with custom reasoners that skip cluster output.
        clusters = list(reasoning.clusters) or aggregate_evidence(
            hits=hit_list,
            documents=doc_list,
        )
        citations = build_citations(clusters)
        cluster_links = link_clusters_to_citations(clusters, citations)

        logger.info(
            "report generation started intent=%s tasks=%d clusters=%d citations=%d",
            plan.intent.value,
            len(plan.tasks),
            len(clusters),
            len(citations),
        )

        sections: list[ReportSection] = [
            self._plan_section(plan),
            self._summary_section(reasoning, cluster_links),
        ]

        findings = self._findings_section(clusters, cluster_links)
        if findings is not None:
            sections.append(findings)

        evidence = self._evidence_section(clusters)
        if evidence is not None:
            sections.append(evidence)

        conflict_section = self._conflicts_section(reasoning, reflection)
        if conflict_section is not None:
            sections.append(conflict_section)

        if reflection is not None:
            sections.append(self._reflection_section(reflection))

        confidence = reflection.confidence if reflection is not None else reasoning.confidence
        query = plan.query or reasoning.query
        summary = reasoning.answer.strip() or (
            "No supporting evidence was gathered for this query."
        )

        report = ResearchReport(
            query=query,
            title=_title_from_query(query),
            summary=summary,
            sections=sections,
            citations=citations,
            confidence=confidence,
            metadata={
                "intent": plan.intent.value,
                "task_count": len(plan.tasks),
                "hit_count": len(hit_list),
                "document_count": len(doc_list),
                "cluster_count": len(clusters),
                "conflict_count": len(reasoning.conflicts),
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

    @staticmethod
    def _plan_section(plan: ResearchPlan) -> ReportSection:
        lines = [f"- {step}" for step in plan.steps] or ["- (no plan steps recorded)"]
        return ReportSection(heading="Research Plan", body="\n".join(lines))

    @staticmethod
    def _summary_section(
        reasoning: ReasoningResult,
        cluster_links: Sequence[Sequence[int]],
    ) -> ReportSection:
        body = reasoning.answer.strip() or ("No supporting evidence was gathered for this query.")
        # The summary references every citation used by the top clusters.
        indexes: list[int] = []
        seen: set[int] = set()
        for links in cluster_links:
            for index in links:
                if index in seen:
                    continue
                seen.add(index)
                indexes.append(index)
        return ReportSection(
            heading="Summary",
            body=body,
            citation_indexes=indexes,
        )

    @staticmethod
    def _findings_section(
        clusters: Sequence[EvidenceCluster],
        cluster_links: Sequence[Sequence[int]],
    ) -> ReportSection | None:
        if not clusters:
            return None
        lines: list[str] = []
        for cluster, links in zip(clusters, cluster_links, strict=True):
            refs = " ".join(f"[{index}]" for index in links)
            suffix = f" {refs}" if refs else ""
            lines.append(f"- {cluster.claim}{suffix}")
        # Attach the union of link indexes so downstream renderers can
        # surface "references: ..." blocks without re-walking clusters.
        aggregated = sorted({index for links in cluster_links for index in links})
        return ReportSection(
            heading="Key Findings",
            body="\n".join(lines),
            citation_indexes=aggregated,
        )

    @staticmethod
    def _evidence_section(clusters: Sequence[EvidenceCluster]) -> ReportSection | None:
        if not clusters:
            return None
        parts: list[str] = []
        for cluster in clusters:
            parts.append(f"**{cluster.claim}**")
            for evidence in cluster.evidence:
                title = evidence.title or evidence.source_id
                link = f" ({evidence.url})" if evidence.url else ""
                parts.append(f"- {title}{link}: {evidence.snippet}")
            parts.append("")
        return ReportSection(heading="Evidence", body="\n".join(parts).rstrip())

    @staticmethod
    def _conflicts_section(
        reasoning: ReasoningResult,
        reflection: ReflectionResult | None,
    ) -> ReportSection | None:
        conflicts = list(reasoning.conflicts)
        if not conflicts and reflection is not None:
            conflicts = list(reflection.conflicts)
        if not conflicts:
            return None
        lines: list[str] = []
        for index, conflict in enumerate(conflicts, start=1):
            lines.extend(
                [
                    f"{index}. {conflict.reason}",
                    f"   - Claim A: {conflict.claim_a}",
                    f"   - Claim B: {conflict.claim_b}",
                ]
            )
        return ReportSection(heading="Conflicts", body="\n".join(lines))

    @staticmethod
    def _reflection_section(reflection: ReflectionResult) -> ReportSection:
        lines = [
            f"Sufficient: {'yes' if reflection.is_sufficient else 'no'}",
            f"Confidence: {reflection.confidence:.2f}",
        ]
        if reflection.gaps:
            lines.append("")
            lines.append("Gaps:")
            lines.extend(f"- {gap}" for gap in reflection.gaps)
        if reflection.follow_up_queries:
            lines.append("")
            lines.append("Suggested follow-ups:")
            lines.extend(f"- {query}" for query in reflection.follow_up_queries)
        return ReportSection(heading="Reflection", body="\n".join(lines))


__all__ = ["SimpleReportGenerator"]
