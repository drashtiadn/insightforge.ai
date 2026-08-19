"""Request and response models for the research API."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ResearchRequest(BaseModel):
    """Body for ``POST /api/v1/research``."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "What are the trade-offs of hybrid RAG versus pure vector search?",
                    "max_steps": 2,
                    "stub_search": True,
                }
            ]
        }
    )

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Research question to investigate.",
        examples=["What are the trade-offs of hybrid RAG versus pure vector search?"],
    )
    max_steps: int | None = Field(
        default=None,
        ge=1,
        le=12,
        description="Cap how many search tasks run. Omit to run every planned task.",
    )
    stub_search: bool = Field(
        default=False,
        description=(
            "Use offline example sources instead of live web search. "
            "Ignored when APP_ENV=production."
        ),
    )

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be empty")
        return cleaned


class ResearchSource(BaseModel):
    """A source cited in the research report."""

    title: str
    url: str


class ResearchResponse(BaseModel):
    """Result of a completed research run."""

    query: str
    report: str = Field(description="Markdown research report.")
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    phase: str
    ok: bool = Field(description="True when the run finished without recorded errors.")
    errors: list[str] = Field(default_factory=list)
    transitions: list[str] = Field(default_factory=list)
    sources: list[ResearchSource] = Field(default_factory=list)
