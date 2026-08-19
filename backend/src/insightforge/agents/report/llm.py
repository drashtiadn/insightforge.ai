"""Gemini-backed report generator with heuristic fallback."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from insightforge.agents.planner.schemas import ResearchPlan
from insightforge.agents.report.base import ReportGenerator
from insightforge.agents.report.simple import SimpleReportGenerator, _title_from_query
from insightforge.core.exceptions import ExternalServiceError
from insightforge.core.logging import get_logger
from insightforge.domain.models import (
    Document,
    ReasoningResult,
    ReflectionResult,
    ReportSection,
    ResearchReport,
    RetrievalHit,
)
from insightforge.infrastructure.llm import LlmMessage, LlmService, parse_model
from insightforge.prompts.report import REPORT_SYSTEM, REPORT_USER
from insightforge.research import (
    aggregate_evidence,
    build_citations,
    link_clusters_to_citations,
)

logger = get_logger(__name__)


class _LlmReportPayload(BaseModel):
    title: str = ""
    summary: str = ""
    findings: list[str] = Field(default_factory=list)


class LlmReportGenerator(ReportGenerator):
    """Keep citation plumbing; ask Gemini for summary and findings prose."""

    def __init__(
        self,
        llm: LlmService,
        *,
        fallback: ReportGenerator | None = None,
    ) -> None:
        super().__init__()
        self._llm = llm
        self._fallback = fallback or SimpleReportGenerator()

    def generate(
        self,
        *,
        plan: ResearchPlan,
        reasoning: ReasoningResult,
        reflection: ReflectionResult | None = None,
        hits: Sequence[RetrievalHit] | None = None,
        documents: Sequence[Document] | None = None,
    ) -> ResearchReport:
        if not self._llm.available:
            logger.info("llm report unavailable; using heuristic report")
            return self._fallback.generate(
                plan=plan,
                reasoning=reasoning,
                reflection=reflection,
                hits=hits,
                documents=documents,
            )

        clusters = list(reasoning.clusters) or aggregate_evidence(
            hits=list(hits or []),
            documents=list(documents or []),
        )
        citations = build_citations(clusters)
        cluster_links = link_clusters_to_citations(clusters, citations)
        claim_lines: list[str] = []
        for cluster, links in zip(clusters, cluster_links, strict=True):
            refs = " ".join(f"[{index}]" for index in links)
            claim_lines.append(f"- {cluster.claim} {refs}".rstrip())

        messages: list[LlmMessage] = [
            {"role": "system", "content": REPORT_SYSTEM},
            {
                "role": "user",
                "content": REPORT_USER.format(
                    query=plan.query or reasoning.query,
                    intent=plan.intent.value,
                    answer=reasoning.answer or "(none)",
                    claims="\n".join(claim_lines) or "(none)",
                ),
            },
        ]
        try:
            raw = self._llm.complete(messages, json_mode=True)
            payload = parse_model(raw, _LlmReportPayload)
        except ExternalServiceError as exc:
            logger.warning("llm report failed; using heuristic report error=%s", exc)
            return self._fallback.generate(
                plan=plan,
                reasoning=reasoning,
                reflection=reflection,
                hits=hits,
                documents=documents,
            )

        fallback_report = self._fallback.generate(
            plan=plan,
            reasoning=reasoning,
            reflection=reflection,
            hits=hits,
            documents=documents,
        )
        title = payload.title.strip() or fallback_report.title or _title_from_query(plan.query)
        summary = payload.summary.strip() or fallback_report.summary
        findings_body = "\n".join(f"- {item.strip()}" for item in payload.findings if item.strip())
        sections: list[ReportSection] = [
            section
            for section in fallback_report.sections
            if section.heading not in {"Summary", "Key Findings"}
        ]
        summary_section = ReportSection(
            heading="Summary",
            body=summary,
            citation_indexes=sorted({index for links in cluster_links for index in links}),
        )
        rebuilt = [sections[0]] if sections and sections[0].heading == "Research Plan" else []
        rebuilt.append(summary_section)
        if findings_body:
            rebuilt.append(
                ReportSection(
                    heading="Key Findings",
                    body=findings_body,
                    citation_indexes=sorted({index for links in cluster_links for index in links}),
                )
            )
        rebuilt.extend(section for section in sections if section.heading != "Research Plan")
        return fallback_report.model_copy(
            update={
                "title": title,
                "summary": summary,
                "sections": rebuilt,
            }
        )
