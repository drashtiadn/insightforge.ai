"""FastAPI app factory.

Run locally:

    uv run uvicorn insightforge.main:app --reload
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from insightforge.api.middleware import RequestLoggingMiddleware
from insightforge.api.routers.root import router as root_router
from insightforge.api.v1 import api_router
from insightforge.core.config import get_settings
from insightforge.core.exceptions import register_exception_handlers
from insightforge.core.logging import configure_logging


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Startup and shutdown hooks for shared resources (DB, clients, etc.)."""

    yield


def create_app() -> FastAPI:
    """Create the FastAPI app with logging, middleware, and routes wired."""

    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.is_production)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan,
    )
    app.add_middleware(RequestLoggingMiddleware)
    register_exception_handlers(app)
    app.include_router(root_router)
    app.include_router(api_router, prefix=settings.api_v1_prefix)
    return app


app = create_app()
