"""HTTP routers."""

from insightforge.api.routers.health import router as health_router
from insightforge.api.routers.root import router as root_router

__all__ = ["health_router", "root_router"]
