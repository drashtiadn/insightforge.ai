"""FastAPI app factory.

Run locally:

    uv run uvicorn insightforge.main:app --reload
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from insightforge.api.routers.root import router as root_router
from insightforge.api.v1 import api_router
from insightforge.core.config import get_settings


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Startup and shutdown hooks for shared resources (DB, clients, etc.)."""

    # startup
    yield
    # shutdown


def create_app() -> FastAPI:
    """Create and wire the FastAPI application."""

    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan,
    )

    app.include_router(root_router)
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    return app


app = create_app()
