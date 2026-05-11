from __future__ import annotations

import structlog

from nexus.memory.base import MemoryEntry, MemoryScope


class InMemoryStore:
    """Session-scoped in-process MemoryStore implementation."""

    def __init__(self) -> None:
        """Create an empty in-memory store."""
        self._entries: dict[str, MemoryEntry] = {}
        self._logger = structlog.get_logger()

    def store(self, key: str, entry: MemoryEntry) -> None:
        """Store a SESSION entry under a namespaced key."""
        self._ensure_supported_scope(entry.scope)
        storage_key = self._storage_key(key, entry.scope, entry.session_id)
        self._entries[storage_key] = entry
        self._logger.debug(
            "memory entry stored",
            key=key,
            scope=entry.scope.value,
            session_id=entry.session_id,
        )

    def retrieve(
        self,
        key: str,
        scope: MemoryScope,
        session_id: str,
    ) -> MemoryEntry | None:
        """Retrieve a SESSION entry and evict it first when expired."""
        self._ensure_supported_scope(scope)
        storage_key = self._storage_key(key, scope, session_id)
        entry = self._entries.get(storage_key)

        if entry is None:
            self._logger.debug(
                "memory retrieve miss",
                key=key,
                scope=scope.value,
                session_id=session_id,
            )
            return None

        if entry.is_expired():
            del self._entries[storage_key]
            self._logger.debug(
                "memory retrieve miss",
                key=key,
                scope=scope.value,
                session_id=session_id,
            )
            return None

        self._logger.debug(
            "memory retrieve hit",
            key=key,
            scope=scope.value,
            session_id=session_id,
        )
        return entry

    def search(
        self,
        query: str,
        scope: MemoryScope,
        session_id: str,
        top_k: int = 5,
    ) -> list[MemoryEntry]:
        """Return recent non-expired SESSION entries for one session."""
        self._ensure_supported_scope(scope)
        prefix = self._storage_prefix(scope, session_id)
        entries = [
            entry
            for storage_key, entry in self._entries.items()
            if storage_key.startswith(prefix) and not entry.is_expired()
        ]
        return sorted(entries, key=lambda entry: entry.created_at, reverse=True)[:top_k]

    def evict(self, scope: MemoryScope, session_id: str) -> int:
        """Delete expired SESSION entries for one session."""
        self._ensure_supported_scope(scope)
        prefix = self._storage_prefix(scope, session_id)
        expired_keys = [
            storage_key
            for storage_key, entry in self._entries.items()
            if storage_key.startswith(prefix) and entry.is_expired()
        ]
        for storage_key in expired_keys:
            del self._entries[storage_key]

        count = len(expired_keys)
        self._logger.info(
            "memory entries evicted",
            count=count,
            scope=scope.value,
            session_id=session_id,
        )
        return count

    @staticmethod
    def _storage_key(key: str, scope: MemoryScope, session_id: str) -> str:
        """Build the internal namespaced storage key."""
        return f"{scope.value}:{session_id}:{key}"

    @staticmethod
    def _storage_prefix(scope: MemoryScope, session_id: str) -> str:
        """Build the internal namespace prefix for session scans."""
        return f"{scope.value}:{session_id}:"

    @staticmethod
    def _ensure_supported_scope(scope: MemoryScope) -> None:
        """Raise for memory tiers intentionally deferred to Day 6."""
        if scope is MemoryScope.USER:
            raise NotImplementedError("USER scope requires vector backend — see Day 6")
        if scope is MemoryScope.AGENT:
            raise NotImplementedError("AGENT scope requires vector backend — see Day 6")
        if scope is MemoryScope.GLOBAL:
            raise NotImplementedError("GLOBAL scope requires distributed store — see Day 6")
