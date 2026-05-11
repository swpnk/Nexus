from __future__ import annotations

from datetime import UTC, datetime, timedelta
from time import sleep

from nexus.core.agent import AgentContext, AgentResult, BaseAgent
from nexus.memory import InMemoryStore, MemoryEntry, MemoryScope
from nexus.providers.base import LLMProvider


class FakeProvider:
    @property
    def model(self) -> str:
        return "fake-model"

    async def complete(self, prompt: str, **kwargs: object) -> str:
        return prompt

    async def complete_with_system(self, system: str, user: str, **kwargs: object) -> str:
        return user


class MemoryAgent(BaseAgent):
    async def execute(self) -> AgentResult:
        return AgentResult(output="ok", success=True)


def make_entry(
    content: str,
    *,
    scope: MemoryScope = MemoryScope.SESSION,
    agent_id: str = "agent-1",
    session_id: str = "s1",
    created_at: datetime | None = None,
    ttl_seconds: int | None = None,
) -> MemoryEntry:
    return MemoryEntry(
        content=content,
        scope=scope,
        agent_id=agent_id,
        session_id=session_id,
        created_at=created_at or datetime.now(UTC),
        ttl_seconds=ttl_seconds,
    )


def test_session_store_and_retrieve() -> None:
    store = InMemoryStore()
    entry = make_entry("value", session_id="s1")

    store.store("k1", entry)

    retrieved = store.retrieve("k1", MemoryScope.SESSION, "s1")
    assert retrieved is not None
    assert retrieved.content == "value"


def test_session_isolation_between_agents() -> None:
    store = InMemoryStore()
    entry = make_entry("private", session_id="s1")

    store.store("k1", entry)

    assert store.retrieve("k1", MemoryScope.SESSION, "s2") is None


def test_retrieve_nonexistent_key_returns_none() -> None:
    store = InMemoryStore()

    assert store.retrieve("missing", MemoryScope.SESSION, "s1") is None


def test_user_scope_raises_not_implemented() -> None:
    store = InMemoryStore()

    try:
        store.store("k1", make_entry("value", scope=MemoryScope.USER))
    except NotImplementedError as exc:
        assert "USER scope" in str(exc)
    else:
        raise AssertionError("USER scope should raise NotImplementedError")


def test_agent_scope_raises_not_implemented() -> None:
    store = InMemoryStore()

    try:
        store.store("k1", make_entry("value", scope=MemoryScope.AGENT))
    except NotImplementedError as exc:
        assert "AGENT scope" in str(exc)
    else:
        raise AssertionError("AGENT scope should raise NotImplementedError")


def test_global_scope_raises_not_implemented() -> None:
    store = InMemoryStore()

    try:
        store.store("k1", make_entry("value", scope=MemoryScope.GLOBAL))
    except NotImplementedError as exc:
        assert "GLOBAL scope" in str(exc)
    else:
        raise AssertionError("GLOBAL scope should raise NotImplementedError")


def test_expired_entry_returns_none() -> None:
    store = InMemoryStore()
    entry = make_entry("value", ttl_seconds=1)

    store.store("k1", entry)
    sleep(2)

    assert store.retrieve("k1", MemoryScope.SESSION, "s1") is None


def test_non_expired_entry_returns_entry() -> None:
    store = InMemoryStore()
    entry = make_entry("value", ttl_seconds=60)

    store.store("k1", entry)

    assert store.retrieve("k1", MemoryScope.SESSION, "s1") == entry


def test_no_ttl_entry_never_expires() -> None:
    store = InMemoryStore()
    entry = make_entry("value", ttl_seconds=None)

    store.store("k1", entry)

    assert store.retrieve("k1", MemoryScope.SESSION, "s1") == entry


def test_search_returns_session_scoped_entries_only() -> None:
    store = InMemoryStore()
    for index in range(3):
        store.store(f"s1-{index}", make_entry(f"s1-{index}", session_id="s1"))
    for index in range(2):
        store.store(f"s2-{index}", make_entry(f"s2-{index}", session_id="s2"))

    results = store.search("ignored", MemoryScope.SESSION, "s1")

    assert {entry.session_id for entry in results} == {"s1"}
    assert len(results) == 3


def test_search_top_k_limits_results() -> None:
    store = InMemoryStore()
    for index in range(5):
        store.store(f"k{index}", make_entry(f"value-{index}"))

    results = store.search("ignored", MemoryScope.SESSION, "s1", top_k=3)

    assert len(results) == 3


def test_search_returns_most_recent_first() -> None:
    store = InMemoryStore()
    now = datetime.now(UTC)
    oldest = make_entry("oldest", created_at=now - timedelta(minutes=3))
    newest = make_entry("newest", created_at=now)
    middle = make_entry("middle", created_at=now - timedelta(minutes=1))

    store.store("oldest", oldest)
    store.store("newest", newest)
    store.store("middle", middle)

    results = store.search("ignored", MemoryScope.SESSION, "s1")

    assert [entry.content for entry in results] == ["newest", "middle", "oldest"]


def test_evict_removes_expired_entries_only() -> None:
    store = InMemoryStore()
    store.store("expired", make_entry("expired", ttl_seconds=1))
    store.store("fresh", make_entry("fresh", ttl_seconds=60))
    sleep(2)

    count = store.evict(MemoryScope.SESSION, "s1")

    assert count == 1
    assert store.retrieve("expired", MemoryScope.SESSION, "s1") is None
    assert store.retrieve("fresh", MemoryScope.SESSION, "s1") is not None


def test_evict_returns_count_of_evicted() -> None:
    store = InMemoryStore()
    for index in range(3):
        store.store(f"k{index}", make_entry(f"value-{index}", ttl_seconds=1))
    sleep(2)

    assert store.evict(MemoryScope.SESSION, "s1") == 3


def test_base_agent_accepts_memory_store() -> None:
    memory = InMemoryStore()
    agent = MemoryAgent(
        AgentContext(agent_id="agent-1", task="task"),
        FakeProvider(),
        memory=memory,
    )

    assert agent.memory is memory


def test_base_agent_works_without_memory() -> None:
    agent = MemoryAgent(AgentContext(agent_id="agent-1", task="task"), FakeProvider())

    assert agent.memory is None
