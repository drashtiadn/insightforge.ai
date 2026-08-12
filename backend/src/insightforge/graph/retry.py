"""Retry helper for transient workflow failures."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

# Errors that are worth retrying (timeouts, brief network blips, etc.).
RETRYABLE = (TimeoutError, ConnectionError, OSError)


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    retry_on: tuple[type[BaseException], ...] = RETRYABLE,
) -> T:
    """Call ``fn`` up to ``max_attempts`` times on transient errors.

    Raises the last matching exception if every attempt fails.
    Non-retryable exceptions are raised immediately.
    """

    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    last_error: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except retry_on as exc:
            last_error = exc
            if attempt == max_attempts:
                raise
    raise RuntimeError(f"retry failed: {last_error}")
