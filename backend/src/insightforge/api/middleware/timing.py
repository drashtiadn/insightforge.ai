"""Request timing and access-log middleware."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from insightforge.core.logging import get_logger

logger = get_logger(__name__)

PROCESS_TIME_HEADER = "X-Process-Time"

__all__ = ["PROCESS_TIME_HEADER", "RequestTimingMiddleware"]


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """Measure request duration, expose it on the response, and write an access log."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        response.headers[PROCESS_TIME_HEADER] = str(duration_ms)
        logger.info(
            "request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response
