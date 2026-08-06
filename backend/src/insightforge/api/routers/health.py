"""Health check endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends

from insightforge.api.schemas import HealthResponse
from insightforge.core.config import Settings, get_settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Liveness check")
async def health_check(
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthResponse:
    """Return basic app info to show the process is up."""

    return HealthResponse(
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
    )
