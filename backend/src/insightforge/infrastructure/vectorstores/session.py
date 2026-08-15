"""Per-research-session temporary vector stores."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from insightforge.core.exceptions import ValidationFailedError
from insightforge.core.logging import get_logger
from insightforge.infrastructure.vectorstores.stores.memory import MemoryVectorStore

logger = get_logger(__name__)


@dataclass
class _SessionEntry:
    store: MemoryVectorStore
    created_at: float
    last_access: float = field(init=False)

    def __post_init__(self) -> None:
        self.last_access = self.created_at


class SessionVectorStoreRegistry:
    """Map ``session_id`` → temporary ``MemoryVectorStore`` with TTL expiry.

    Each research / chat session gets an isolated in-memory index so retrieved
    chunks do not leak across users. Expired sessions are dropped on access or
    when ``cleanup()`` runs.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = 3600.0,
        dimensions: int | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValidationFailedError(
                "session TTL must be > 0",
                details={"field": "ttl_seconds", "value": ttl_seconds},
            )
        self._ttl_seconds = ttl_seconds
        self._dimensions = dimensions
        self._sessions: dict[str, _SessionEntry] = {}
        self._lock = threading.RLock()

    @property
    def ttl_seconds(self) -> float:
        return self._ttl_seconds

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def get(self, session_id: str) -> MemoryVectorStore:
        """Return the store for ``session_id``, creating it if needed."""

        cleaned = session_id.strip()
        if not cleaned:
            raise ValidationFailedError(
                "session_id must not be empty",
                details={"field": "session_id"},
            )

        now = time.monotonic()
        with self._lock:
            self._drop_expired_unlocked(now)
            entry = self._sessions.get(cleaned)
            if entry is None:
                store = MemoryVectorStore(dimensions=self._dimensions)
                entry = _SessionEntry(store=store, created_at=now)
                self._sessions[cleaned] = entry
                logger.info(
                    "session vector store created session_id=%s ttl=%.1f",
                    cleaned,
                    self._ttl_seconds,
                    extra={"session_id": cleaned, "ttl_seconds": self._ttl_seconds},
                )
            else:
                entry.last_access = now
            return entry.store

    def drop(self, session_id: str) -> bool:
        """Remove one session store. Returns True when it existed."""

        cleaned = session_id.strip()
        if not cleaned:
            return False
        with self._lock:
            entry = self._sessions.pop(cleaned, None)
            if entry is None:
                return False
            entry.store.clear()
            logger.info(
                "session vector store dropped session_id=%s",
                cleaned,
                extra={"session_id": cleaned},
            )
            return True

    def cleanup(self) -> int:
        """Drop expired sessions. Returns the number removed."""

        with self._lock:
            return self._drop_expired_unlocked(time.monotonic())

    def clear(self) -> None:
        """Drop every session store."""

        with self._lock:
            count = len(self._sessions)
            for entry in self._sessions.values():
                entry.store.clear()
            self._sessions.clear()
            logger.info(
                "session vector stores cleared count=%d",
                count,
                extra={"cleared": count},
            )

    def _drop_expired_unlocked(self, now: float) -> int:
        expired = [
            session_id
            for session_id, entry in self._sessions.items()
            if (now - entry.last_access) > self._ttl_seconds
        ]
        for session_id in expired:
            entry = self._sessions.pop(session_id)
            entry.store.clear()
            logger.info(
                "session vector store expired session_id=%s",
                session_id,
                extra={"session_id": session_id},
            )
        return len(expired)
