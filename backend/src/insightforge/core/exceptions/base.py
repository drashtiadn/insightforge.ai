"""Application exception hierarchy.

Business and domain code should raise these exceptions — never
``fastapi.HTTPException`` — so the web layer stays at the edge.
"""

from __future__ import annotations

from typing import Any


class AppException(Exception):
    """Base application error mapped to a structured API response."""

    code: str = "app_error"
    message: str = "An application error occurred"
    status_code: int = 500

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        details: dict[str, Any] | list[Any] | None = None,
    ) -> None:
        self.message = message or type(self).message
        self.code = code or type(self).code
        self.details = details
        super().__init__(self.message)


class BadRequestError(AppException):
    """Request cannot be processed (400)."""

    code = "bad_request"
    message = "Bad request"
    status_code = 400


class UnauthorizedError(AppException):
    """Authentication required or failed (401)."""

    code = "unauthorized"
    message = "Authentication required"
    status_code = 401


class ForbiddenError(AppException):
    """Authenticated but not allowed (403)."""

    code = "forbidden"
    message = "Permission denied"
    status_code = 403


class NotFoundError(AppException):
    """Requested resource does not exist (404)."""

    code = "not_found"
    message = "Resource not found"
    status_code = 404


class ConflictError(AppException):
    """State conflict, e.g. duplicate resource (409)."""

    code = "conflict"
    message = "Conflict"
    status_code = 409


class ValidationFailedError(AppException):
    """Business or input validation failed (422)."""

    code = "validation_failed"
    message = "Validation failed"
    status_code = 422


class ExternalServiceError(AppException):
    """Upstream dependency failed (502)."""

    code = "external_service_error"
    message = "External service error"
    status_code = 502
