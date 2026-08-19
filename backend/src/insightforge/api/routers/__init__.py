"""HTTP routers."""

from insightforge.api.routers.health import router as health_router
from insightforge.api.routers.research import router as research_router
from insightforge.api.routers.root import router as root_router

__all__ = ["health_router", "research_router", "root_router"]
