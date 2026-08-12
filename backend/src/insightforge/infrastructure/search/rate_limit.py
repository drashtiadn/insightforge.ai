"""Per-provider rate limiting for outbound search calls."""

from __future__ import annotations

import threading
import time

from insightforge.core.logging import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """Simple per-key minimum-interval limiter (thread-safe).

    ``calls_per_second=2`` means at most one call every 0.5s for a given key.
    """

    def __init__(self, *, calls_per_second: float = 2.0) -> None:
        if calls_per_second <= 0:
            raise ValueError("calls_per_second must be > 0")
        self._min_interval = 1.0 / calls_per_second
        self._lock = threading.Lock()
        self._next_allowed: dict[str, float] = {}

    @property
    def min_interval(self) -> float:
        return self._min_interval

    def acquire(self, key: str) -> float:
        """Block until ``key`` may proceed. Returns seconds waited."""

        waited = 0.0
        with self._lock:
            now = time.monotonic()
            allowed_at = self._next_allowed.get(key, now)
            if allowed_at > now:
                waited = allowed_at - now
                sleep_for = waited
                self._next_allowed[key] = allowed_at + self._min_interval
            else:
                sleep_for = 0.0
                self._next_allowed[key] = now + self._min_interval

        if sleep_for > 0:
            logger.debug(
                "rate limit wait key=%s waited=%.3fs",
                key,
                sleep_for,
                extra={"rate_limit_key": key, "waited_seconds": sleep_for},
            )
            time.sleep(sleep_for)
        return waited
