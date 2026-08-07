"""Structured logging for the application.

Development uses readable text; production uses JSON lines for log aggregators.
Request IDs come from ``request_id_var`` (set by request middleware).
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, TextIO

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime", "color_message"}

__all__ = [
    "JsonFormatter",
    "configure_logging",
    "get_logger",
    "request_id_var",
]


class JsonFormatter(logging.Formatter):
    """Serialize log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id:
            data["request_id"] = request_id
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                data[key] = value
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        return json.dumps(data, default=str)


class _TextFormatter(logging.Formatter):
    """Human-readable format with optional request id prefix."""

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        request_id = request_id_var.get()
        if request_id:
            return f"[{request_id}] {line}"
        return line


class _AppStreamHandler(logging.StreamHandler[TextIO]):
    """Stdout handler owned by ``configure_logging`` (safe to replace on reconfigure)."""


def configure_logging(level: str = "INFO", *, json_logs: bool = False) -> None:
    """Configure root logging once per process.

    Replaces only handlers created by this function so test capture handlers stay intact.
    Quiets uvicorn access logs because request middleware owns access logging.
    """

    root = logging.getLogger()
    root.setLevel(level.upper())

    for handler in list(root.handlers):
        if isinstance(handler, _AppStreamHandler):
            root.removeHandler(handler)

    handler = _AppStreamHandler(sys.stdout)
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(_TextFormatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    root.addHandler(handler)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (use ``__name__`` at the call site)."""

    return logging.getLogger(name)
