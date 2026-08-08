"""Middleware registration for the FastAPI application."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from insightforge.api.middleware.request_id import RequestIdMiddleware
from insightforge.api.middleware.timing import RequestTimingMiddleware
from insightforge.core.config import Settings

__all__ = ["register_middleware"]


def register_middleware(app: FastAPI, settings: Settings) -> None:
    """Attach middleware. Last added runs first on the request path.

    Order on the way in: CORS → request id → timing → routes.
    """

    app.add_middleware(RequestTimingMiddleware)
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
        expose_headers=settings.cors_expose_headers,
    )
