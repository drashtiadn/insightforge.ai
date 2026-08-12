"""Structured outputs produced by the research planner."""

from __future__ import annotations

from pydantic import BaseModel, Field

from insightforge.shared.enums import QueryIntent, SearchProviderHint


class QueryAnalysis(BaseModel):
    """Normalized view of the raw user query."""

    original_query: str
    normalized_query: str
    keywords: list[str] = Field(default_factory=list)
    token_count: int = Field(ge=0)


class ResearchTask(BaseModel):
    """One searchable unit of work for later provider execution."""

    id: str
    description: str
    search_query: str
    providers: list[SearchProviderHint] = Field(default_factory=list)
    priority: int = Field(ge=1, description="Lower number = higher priority")


class ResearchPlan(BaseModel):
    """Full planner result: analysis, intent, and ordered tasks."""

    query: str
    analysis: QueryAnalysis
    intent: QueryIntent
    tasks: list[ResearchTask] = Field(default_factory=list)

    @property
    def steps(self) -> list[str]:
        """Human-readable plan lines for reports and legacy graph state."""

        return [task.description for task in self.tasks]
