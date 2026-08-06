"""Version 1 API routes, mounted under `/api/v1`."""

from fastapi import APIRouter

from insightforge.api.routers import health_router

api_router = APIRouter()
api_router.include_router(health_router)

__all__ = ["api_router"]
