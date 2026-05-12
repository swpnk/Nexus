from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Protocol, runtime_checkable


class MemoryScope(StrEnum):
    """Trust boundary that determines where a memory entry is visible."""

    SESSION = "session"
    USER = "user"
    AGENT = "agent"
    GLOBAL = "global"


@dataclass(frozen=True)
class MemoryEntry:
    """Stored memory value with mandatory scope and writer metadata."""

    content: str
    scope: MemoryScope
    agent_id: str
    session_id: str
    created_at: datetime
    ttl_seconds: int | None = None

    def __post_init__(self) -> None:
        """Validate that created_at satisfies the framework UTC invariant."""
        if self.created_at.tzinfo is None or self.created_at.utcoffset() != timedelta(0):
            raise ValueError("created_at must be timezone-aware UTC")

    def is_expired(self) -> bool:
        """Return True when this entry's TTL has elapsed."""
        if self.ttl_seconds is None:
            return False
        return datetime.now(UTC) - self.created_at > timedelta(seconds=self.ttl_seconds)


@runtime_checkable
class MemoryStore(Protocol):
    """Structural contract for scoped memory backends."""

    def store(self, key: str, entry: MemoryEntry) -> None:
        """Persist a memory entry under a caller-provided key."""
        ...

    def retrieve(
        self,
        key: str,
        scope: MemoryScope,
        session_id: str,
    ) -> MemoryEntry | None:
        """Return a scoped entry by exact key, or None when absent or expired."""
        ...

    def search(
        self,
        query: str,
        scope: MemoryScope,
        session_id: str,
        top_k: int = 5,
    ) -> list[MemoryEntry]:
        """Return up to top_k scoped entries matching the backend search policy."""
        ...

    def evict(self, scope: MemoryScope, session_id: str) -> int:
        """Remove expired entries for a scope and return the removal count."""
        ...
