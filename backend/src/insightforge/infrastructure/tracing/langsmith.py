"""LangSmith setup, tracing, and debug helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from insightforge.core.config import Settings, get_settings
from insightforge.core.logging import get_logger

logger = get_logger(__name__)

# Env vars LangGraph / LangChain read for automatic tracing.
_ENV_TRACING = "LANGSMITH_TRACING"
_ENV_API_KEY = "LANGSMITH_API_KEY"
_ENV_PROJECT = "LANGSMITH_PROJECT"
_ENV_ENDPOINT = "LANGSMITH_ENDPOINT"


@dataclass(frozen=True)
class TracingStatus:
    """Current LangSmith configuration (safe to log / return from health)."""

    enabled: bool
    project: str
    endpoint: str
    has_api_key: bool
    reason: str


def configure_tracing(settings: Settings | None = None) -> TracingStatus:
    """Apply LangSmith settings to the process environment.

    Call once at app startup. When tracing is off or the API key is missing,
    tracing stays disabled and the workflow still runs normally.
    """

    cfg = settings or get_settings()

    if not cfg.langsmith_tracing:
        os.environ[_ENV_TRACING] = "false"
        status = TracingStatus(
            enabled=False,
            project=cfg.langsmith_project,
            endpoint=cfg.langsmith_endpoint,
            has_api_key=cfg.langsmith_api_key is not None,
            reason="LANGSMITH_TRACING is false",
        )
        logger.info(
            "LangSmith tracing disabled (%s)",
            status.reason,
        )
        return status

    if cfg.langsmith_api_key is None:
        os.environ[_ENV_TRACING] = "false"
        status = TracingStatus(
            enabled=False,
            project=cfg.langsmith_project,
            endpoint=cfg.langsmith_endpoint,
            has_api_key=False,
            reason="LANGSMITH_API_KEY is missing",
        )
        logger.warning(
            "LangSmith tracing requested but API key is missing (project=%s)",
            cfg.langsmith_project,
        )
        return status

    os.environ[_ENV_TRACING] = "true"
    os.environ[_ENV_API_KEY] = cfg.langsmith_api_key.get_secret_value()
    os.environ[_ENV_PROJECT] = cfg.langsmith_project
    os.environ[_ENV_ENDPOINT] = cfg.langsmith_endpoint

    status = TracingStatus(
        enabled=True,
        project=cfg.langsmith_project,
        endpoint=cfg.langsmith_endpoint,
        has_api_key=True,
        reason="configured",
    )
    logger.info(
        "LangSmith tracing enabled (project=%s endpoint=%s)",
        status.project,
        status.endpoint,
    )
    return status


def tracing_status() -> TracingStatus:
    """Read the effective tracing status from the current environment."""

    enabled = os.environ.get(_ENV_TRACING, "false").lower() == "true"
    has_key = bool(os.environ.get(_ENV_API_KEY))
    project = os.environ.get(_ENV_PROJECT, "insightforge")
    endpoint = os.environ.get(_ENV_ENDPOINT, "https://api.smith.langchain.com")

    if enabled and has_key:
        reason = "configured"
    elif enabled and not has_key:
        reason = "LANGSMITH_API_KEY is missing"
        enabled = False
    else:
        reason = "LANGSMITH_TRACING is false"

    return TracingStatus(
        enabled=enabled and has_key,
        project=project,
        endpoint=endpoint,
        has_api_key=has_key,
        reason=reason,
    )


def debug_hints() -> dict[str, str]:
    """Short hints for inspecting runs in the LangSmith UI."""

    status = tracing_status()
    return {
        "ui": "https://smith.langchain.com",
        "project": status.project,
        "run_name": "run_research",
        "tip": "Open the project, filter by run name run_research, inspect node spans",
    }


def debug_run_summary(
    *,
    phase: str,
    ok: bool,
    score: float,
    errors: tuple[str, ...] | list[str],
    transitions: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """Compact summary to attach as LangSmith metadata or local debug logs."""

    return {
        "phase": phase,
        "ok": ok,
        "score": score,
        "error_count": len(errors),
        "errors": list(errors),
        "transitions": list(transitions),
    }
