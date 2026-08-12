"""Retry helper for transient workflow failures."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

import httpx

T = TypeVar("T")

# Errors that are worth retrying (timeouts, brief network blips, etc.).
RETRYABLE = (
    TimeoutError,
    ConnectionError,
    OSError,
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)


def _backoff_delay(
    attempt: int,
    *,
    base_delay: float,
    max_delay: float,
) -> float:
    """Exponential backoff with full jitter for attempt ``attempt`` (1-based)."""

    exponential = min(max_delay, base_delay * (2 ** (attempt - 1)))
    # Full jitter: uniform in [0, exponential] to avoid synchronized retries.
    return random.uniform(0.0, exponential)


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    retry_on: tuple[type[BaseException], ...] = RETRYABLE,
    base_delay: float = 0.05,
    max_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``fn`` up to ``max_attempts`` times on transient errors.

    Waits with exponential backoff and full jitter between attempts.
    Raises the last matching exception if every attempt fails.
    Non-retryable exceptions are raised immediately.
    """

    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    if base_delay < 0:
        raise ValueError("base_delay must be >= 0")
    if max_delay < 0:
        raise ValueError("max_delay must be >= 0")

    last_error: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except retry_on as exc:
            last_error = exc
            if attempt == max_attempts:
                raise
            delay = _backoff_delay(
                attempt,
                base_delay=base_delay,
                max_delay=max_delay,
            )
            if delay > 0:
                sleep(delay)
    raise RuntimeError(f"retry failed: {last_error}")
