"""Domain models for multi-agent research outputs (Phase 6).

These models are the contracts exchanged between the reasoning, reflection,
and report-generation agents. They stay free of infrastructure imports so
future LLM-backed implementations can populate them without changes here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from insightforge.domain.models.document import Citation

__all__ = [
    "ReasoningResult",
    "ReflectionResult",
    "ReportSection",
    "ResearchReport",
]


class ReasoningResult(BaseModel):
    """Structured answer synthesized by the reasoner agent.

    The reasoner receives ranked evidence (retrieval hits and/or documents)
    and produces a short answer plus key supporting points. Downstream
    agents (reflection, report) consume this to shape their outputs.
    """

    query: str
    answer: str = ""
    key_points: list[str] = Field(default_factory=list)
    used_source_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReflectionResult(BaseModel):
    """Reflection agent's verdict on the current research state.

    ``is_sufficient`` gates whether the workflow may proceed to report
    generation. When False, ``follow_up_queries`` suggests concrete
    additional searches the planner or search agent can execute.
    """

    is_sufficient: bool
    gaps: list[str] = Field(default_factory=list)
    follow_up_queries: list[str] = Field(default_factory=list)
    reasoning: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ReportSection(BaseModel):
    """One heading + body block of a research report."""

    heading: str
    body: str
    citation_indexes: list[int] = Field(default_factory=list)


class ResearchReport(BaseModel):
    """Final artifact produced by the report generator agent."""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    query: str
    title: str
    summary: str = ""
    sections: list[ReportSection] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_markdown(self) -> str:
        """Render the report as a Markdown string."""

        lines: list[str] = [f"# {self.title}", ""]
        if self.summary:
            lines.extend([self.summary, ""])
        for section in self.sections:
            lines.append(f"## {section.heading}")
            lines.append("")
            lines.append(section.body)
            if section.citation_indexes:
                refs = ", ".join(f"[{index}]" for index in section.citation_indexes)
                lines.append("")
                lines.append(f"References: {refs}")
            lines.append("")
        if self.citations:
            lines.append("## Citations")
            lines.append("")
            for index, citation in enumerate(self.citations, start=1):
                formatted = citation.format() or citation.url or citation.title or ""
                lines.append(f"[{index}] {formatted}")
        return "\n".join(lines).rstrip() + "\n"
