"""Health and root response models."""

from pydantic import BaseModel, Field

from insightforge.core.config import Environment


class HealthResponse(BaseModel):
    """Response for `GET /api/v1/health`."""

    status: str = Field(default="ok", description="Overall service status.")
    app_name: str
    version: str
    environment: Environment


class RootResponse(BaseModel):
    """Response for `GET /`."""

    name: str
    version: str
    environment: Environment
    docs_url: str = "/docs"
    health_url: str = "/api/v1/health"
    research_url: str = "/api/v1/research"
