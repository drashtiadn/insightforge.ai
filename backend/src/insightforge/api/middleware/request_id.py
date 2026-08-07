"""Request correlation id middleware."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from insightforge.core.logging import request_id_var

REQUEST_ID_HEADER = "X-Request-ID"

__all__ = ["REQUEST_ID_HEADER", "RequestIdMiddleware"]


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Bind a request id for the request lifetime and echo it on the response.

    Reuses an incoming ``X-Request-ID`` when present; otherwise generates a UUID.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        token = request_id_var.set(request_id)
        try:
            response = await call_next(request)
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            request_id_var.reset(token)
