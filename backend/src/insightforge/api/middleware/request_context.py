"""Request-scoped values for logging (safe with BaseHTTPMiddleware)."""

from __future__ import annotations

from typing import Any

from starlette.requests import Request

_SCOPE_REQUEST_ID = "insightforge_request_id"
_SCOPE_CONTEXT = "insightforge_context"


def bind_request_context(request: Request, request_id: str) -> dict[str, Any]:
    """Attach request id + mutable context bag to the ASGI scope."""

    context: dict[str, Any] = {}
    request.scope[_SCOPE_REQUEST_ID] = request_id
    request.scope[_SCOPE_CONTEXT] = context
    return context


def get_scope_request_id(request: Request) -> str | None:
    value = request.scope.get(_SCOPE_REQUEST_ID)
    return value if isinstance(value, str) else None


def get_scope_context(request: Request) -> dict[str, Any] | None:
    value = request.scope.get(_SCOPE_CONTEXT)
    return value if isinstance(value, dict) else None


def set_user_id(request: Request, user_id: str) -> None:
    """Attach the authenticated user id to the current request (optional)."""

    context = get_scope_context(request)
    if context is not None:
        context["user_id"] = user_id


def get_user_id(request: Request) -> str | None:
    context = get_scope_context(request)
    if context is None:
        return None
    value = context.get("user_id")
    return value if isinstance(value, str) else None


def set_request_error(request: Request, exc: BaseException) -> None:
    """Stash an exception so the access log can include a stack trace."""

    context = get_scope_context(request)
    if context is not None:
        context["error"] = exc


def get_request_error(request: Request) -> BaseException | None:
    context = get_scope_context(request)
    if context is None:
        return None
    value = context.get("error")
    return value if isinstance(value, BaseException) else None
