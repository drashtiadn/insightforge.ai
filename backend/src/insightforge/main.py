"""FastAPI app factory.

Run locally:

    uv run uvicorn insightforge.main:app --reload
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from insightforge.api.middleware import register_middleware
from insightforge.api.routers.root import router as root_router
from insightforge.api.v1 import api_router
from insightforge.core.config import get_settings
from insightforge.core.exceptions import register_exception_handlers
from insightforge.core.logging import configure_logging
from insightforge.infrastructure.tracing import configure_tracing


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Startup and shutdown hooks for shared resources (DB, clients, etc.)."""

    configure_tracing()
    yield


def create_app() -> FastAPI:
    """Create the FastAPI app with logging, middleware, and routes wired."""

    settings = get_settings()
    # DEBUG=true in settings makes logs noisier; FastAPI debug stays off so
    # clients always get structured JSON errors (never raw tracebacks).
    log_level = "DEBUG" if settings.debug else settings.log_level
    configure_logging(log_level, json_logs=settings.is_production)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Multi-agent research API. Use **POST /api/v1/research** to run the LangGraph pipeline."
        ),
        debug=False,
        lifespan=lifespan,
        openapi_tags=[
            {"name": "research", "description": "Run the research pipeline."},
            {"name": "health", "description": "Liveness checks."},
            {"name": "root", "description": "Service discovery."},
        ],
    )
    register_middleware(app, settings)
    register_exception_handlers(app)
    app.include_router(root_router)
    app.include_router(api_router, prefix=settings.api_v1_prefix)
    return app


app = create_app()
