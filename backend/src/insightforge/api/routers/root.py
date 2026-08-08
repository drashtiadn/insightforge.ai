"""Root discovery endpoint (unversioned)."""

from typing import Annotated

from fastapi import APIRouter, Depends

from insightforge.api.schemas import RootResponse
from insightforge.core.config import Settings, get_settings

router = APIRouter(tags=["root"])


@router.get("/", response_model=RootResponse, summary="Service discovery")
async def root(
    settings: Annotated[Settings, Depends(get_settings)],
) -> RootResponse:
    """Return app name, version, and useful links."""

    return RootResponse(
        name=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
        health_url=f"{settings.api_v1_prefix}/health",
    )
