"""FastAPI exception handlers that map errors to ``ErrorResponse``."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from insightforge.api.schemas.errors import ErrorResponse
from insightforge.core.exceptions.base import AppException
from insightforge.core.logging import get_logger, request_id_var

logger = get_logger(__name__)

_STATUS_CODES: dict[int, str] = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    422: "validation_failed",
    429: "too_many_requests",
    500: "internal_error",
    502: "bad_gateway",
    503: "service_unavailable",
}


def _error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | list[Any] | None = None,
) -> JSONResponse:
    """Build a JSON error response.

    Uses ``mode="json"`` so common detail values (UUID, datetime, Decimal, bytes)
    serialize cleanly. If details still cannot be encoded, they are dropped rather
    than turning a handled application error into an unhandled 500.
    """

    request_id = request_id_var.get()
    body = ErrorResponse(
        code=code,
        message=message,
        details=details,
        request_id=request_id,
    )
    try:
        content = body.model_dump(mode="json", exclude_none=True)
    except Exception:
        logger.exception(
            "failed to serialize error details; returning error without details",
            extra={"error_code": code, "status_code": status_code},
        )
        content = ErrorResponse(
            code=code,
            message=message,
            details=None,
            request_id=request_id,
        ).model_dump(mode="json", exclude_none=True)
    return JSONResponse(status_code=status_code, content=content)


async def app_exception_handler(_request: Request, exc: AppException) -> JSONResponse:
    """Handle known application errors raised by domain/application code."""

    log = logger.warning if exc.status_code < 500 else logger.error
    log(
        "application error",
        extra={
            "error_code": exc.code,
            "status_code": exc.status_code,
            "error_message": exc.message,
        },
    )
    return _error_response(
        status_code=exc.status_code,
        code=exc.code,
        message=exc.message,
        details=exc.details,
    )


async def http_exception_handler(
    _request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    """Normalize framework ``HTTPException`` into the standard error schema."""

    message = exc.detail if isinstance(exc.detail, str) else "HTTP error"
    details = None if isinstance(exc.detail, str) else {"detail": exc.detail}
    return _error_response(
        status_code=exc.status_code,
        code=_STATUS_CODES.get(exc.status_code, "http_error"),
        message=message,
        details=details,
    )


async def validation_exception_handler(
    _request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle FastAPI/Pydantic request validation failures."""

    details = [
        {
            "field": ".".join(str(part) for part in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        }
        for error in exc.errors()
    ]
    return _error_response(
        status_code=422,
        code="validation_failed",
        message="Request validation failed",
        details=details,
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Log unexpected failures; never leak internals to the client."""

    logger.exception(
        "unhandled error",
        extra={"method": request.method, "path": request.url.path},
    )
    return _error_response(
        status_code=500,
        code="internal_error",
        message="Internal server error",
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach global exception handlers to the application."""

    app.add_exception_handler(AppException, app_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)
