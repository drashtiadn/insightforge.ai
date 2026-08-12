"""Domain contract for a searchable unit of research work."""

from __future__ import annotations

from pydantic import BaseModel, Field

from insightforge.shared.enums import SearchProviderHint


class ResearchTask(BaseModel):
    """One searchable unit of work for later provider execution.

    Shared across planner output, graph state, and search infrastructure so
    lower layers do not depend on the planner package.
    """

    id: str
    description: str
    search_query: str
    providers: list[SearchProviderHint] = Field(default_factory=list)
    priority: int = Field(ge=1, description="Lower number = higher priority")
