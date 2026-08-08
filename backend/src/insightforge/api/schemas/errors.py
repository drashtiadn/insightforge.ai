"""Standard API error response schema."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Consistent error body returned by all exception handlers."""

    code: str = Field(description="Machine-readable error code.")
    message: str = Field(description="Human-readable error summary.")
    details: dict[str, Any] | list[Any] | None = Field(
        default=None,
        description="Optional structured context (fields, reasons, etc.).",
    )
    request_id: str | None = Field(
        default=None,
        description="Correlation id from X-Request-ID when available.",
    )
