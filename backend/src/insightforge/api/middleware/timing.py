"""Request timing and access-log middleware."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from insightforge.api.middleware.request_context import (
    get_request_error,
    get_scope_request_id,
    get_user_id,
)
from insightforge.core.logging import get_logger, request_id_var

logger = get_logger("insightforge.api")

PROCESS_TIME_HEADER = "X-Process-Time"

__all__ = ["PROCESS_TIME_HEADER", "RequestTimingMiddleware"]


def _request_target(request: Request) -> str:
    """Path plus query string, e.g. ``/api/v1/health`` or ``/search?q=ai``."""

    target = request.url.path
    if request.url.query:
        return f"{target}?{request.url.query}"
    return target


def _client_ip(request: Request) -> str:
    if request.client is None:
        return "-"
    return request.client.host


def _unwrap_error(exc: BaseException | None) -> BaseException | None:
    if exc is None:
        return None
    if isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        return exc.exceptions[0]
    return exc


def _access_fields(
    request: Request,
    *,
    status_code: int,
    duration_ms: float,
    client_ip: str,
) -> dict[str, Any]:
    """Structured fields for production log aggregators."""

    fields: dict[str, Any] = {
        "request_id": get_scope_request_id(request) or request_id_var.get(),
        "method": request.method,
        "path": request.url.path,
        "status_code": status_code,
        "duration_ms": duration_ms,
        "client_ip": client_ip,
    }
    user_id = get_user_id(request)
    if user_id is not None:
        fields["user_id"] = user_id
    return fields


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """Measure request duration, expose it on the response, and write an access log."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start = time.perf_counter()
        response: Response | None = None
        raised: BaseException | None = None
        try:
            response = await call_next(request)
        except BaseException as exc:
            raised = exc
            raise
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            status_code = response.status_code if response is not None else 500
            if response is not None:
                response.headers[PROCESS_TIME_HEADER] = str(duration_ms)

            method = request.method
            target = _request_target(request)
            client_ip = _client_ip(request)
            extra = _access_fields(
                request,
                status_code=status_code,
                duration_ms=duration_ms,
                client_ip=client_ip,
            )
            message = "API %s %s -> %s (%.2fms)"
            args = (method, target, status_code, duration_ms)
            error = _unwrap_error(get_request_error(request) or raised)

            if status_code >= 500 and error is not None:
                logger.error(message, *args, exc_info=error, extra=extra)
            elif status_code >= 500:
                logger.error(message, *args, extra=extra)
            elif status_code >= 400:
                logger.warning(message, *args, extra=extra)
            elif response is not None:
                logger.info(message, *args, extra=extra)

        assert response is not None
        return response
