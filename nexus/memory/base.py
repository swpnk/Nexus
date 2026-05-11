from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Protocol, runtime_checkable


class MemoryScope(StrEnum):
    SESSION = "session"
    USER = "user"
    AGENT = "agent"
    GLOBAL = "global"


@dataclass(frozen=True)
class MemoryEntry:
    content: str
    scope: MemoryScope
    agent_id: str
    session_id: str
    created_at: datetime
    ttl_seconds: int | None = None

    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        return datetime.now(UTC) - self.created_at > timedelta(seconds=self.ttl_seconds)


@runtime_checkable
class MemoryStore(Protocol):
    def store(self, key: str, entry: MemoryEntry) -> None: ...

    def retrieve(
        self,
        key: str,
        scope: MemoryScope,
        session_id: str,
    ) -> MemoryEntry | None: ...

    def search(
        self,
        query: str,
        scope: MemoryScope,
        session_id: str,
        top_k: int = 5,
    ) -> list[MemoryEntry]: ...

    def evict(self, scope: MemoryScope, session_id: str) -> int: ...
