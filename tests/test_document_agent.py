from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from nexus.agents.document_agent import ChunkingConfig, DocumentAgent, DocumentChunk
from nexus.core.agent import AgentContext
from nexus.core.circuit_breaker import BreakerState, CircuitBreaker
from nexus.memory import InMemoryStore, MemoryScope


class FakeProvider:
    def __init__(self, response: str = "Answer grounded in the document.") -> None:
        self.response = response

    @property
    def model(self) -> str:
        return "fake-model"

    async def complete(self, prompt: str, **kwargs: object) -> str:
        return self.response

    async def complete_with_system(self, system: str, user: str, **kwargs: object) -> str:
        return self.response


def make_chunk(content: str = "hello world content") -> DocumentChunk:
    words = content.split()
    if len(words) < 64:
        content = " ".join([*words, *["token"] * (64 - len(words))])
    return DocumentAgent._make_chunk(
        content=content,
        section_path=["Introduction"],
        page_number=1,
        position_in_section=0,
    )


def make_context(
    content: bytes,
    *,
    query: str = "what is this about",
    session_id: str = "s1",
    chunking_config: dict[str, Any] | None = None,
) -> AgentContext:
    config: dict[str, Any] = {
        "content": content,
        "query": query,
        "session_id": session_id,
    }
    if chunking_config is not None:
        config["chunking_config"] = chunking_config
    return AgentContext(agent_id="document-agent-1", task=query, config=config)


def sample_pages() -> list[tuple[int, str, list[str]]]:
    long_para = " ".join(["nexus"] * 80)
    return [
        (1, f"{long_para}\n\n{long_para}", ["INTRODUCTION"]),
    ]


@pytest.mark.asyncio
async def test_chunking_respects_paragraph_boundaries() -> None:
    agent = DocumentAgent(make_context(b"%PDF"), FakeProvider())
    config = ChunkingConfig(min_tokens_per_chunk=5, max_tokens_per_chunk=200)
    pages = [
        (
            1,
            "Alpha paragraph one has many words here for testing.\n\n"
            "Beta paragraph two also has enough words here for testing.",
            [],
        )
    ]

    chunks = agent._chunk_document(pages, config)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert "\n\n" not in chunk.content


def test_minimum_chunk_size_discards_tiny_chunks() -> None:
    agent = DocumentAgent(make_context(b"%PDF"), FakeProvider())
    config = ChunkingConfig(min_tokens_per_chunk=10, max_tokens_per_chunk=500)
    pages = [
        (
            1,
            "Short.\n\nThis paragraph has well over ten tokens in it for sure here today.",
            [],
        )
    ]

    chunks = agent._chunk_document(pages, config)

    assert chunks
    assert all(chunk.token_count >= 10 for chunk in chunks)
    assert len(chunks) == 1


def test_lost_in_middle_two_chunks() -> None:
    agent = DocumentAgent(make_context(b"%PDF"), FakeProvider())
    chunk_a = make_chunk("alpha " * 70)
    chunk_b = make_chunk("beta " * 70)
    scored = [(0.9, chunk_a), (0.6, chunk_b)]

    assembled = agent._assemble_context([], scored)

    assert assembled[0] is chunk_a
    assert assembled[-1] is chunk_b


def test_lost_in_middle_five_chunks() -> None:
    agent = DocumentAgent(make_context(b"%PDF"), FakeProvider())
    chunk_a = make_chunk("alpha " * 70)
    chunk_b = make_chunk("beta " * 70)
    chunk_c = make_chunk("gamma " * 70)
    chunk_d = make_chunk("delta " * 70)
    chunk_e = make_chunk("epsilon " * 70)
    scored = [(0.9, chunk_a), (0.8, chunk_b), (0.7, chunk_c), (0.6, chunk_d), (0.5, chunk_e)]

    assembled = agent._assemble_context([], scored)

    assert assembled[0] is chunk_a
    assert assembled[-1] is chunk_b
    assert assembled[1:-1] == [chunk_c, chunk_d, chunk_e]


@pytest.mark.asyncio
async def test_cache_hit_returns_cached_chunk() -> None:
    memory = InMemoryStore()
    agent = DocumentAgent(make_context(b"%PDF"), FakeProvider(), memory=memory)
    chunks = [make_chunk("cached chunk content for memory testing purposes here")]

    await agent._get_or_cache_chunks(chunks, session_id="s1", doc_hash="abc")
    first_entry = memory.retrieve(
        key=f"doc_chunk:abc:{chunks[0].chunk_id}",
        scope=MemoryScope.SESSION,
        session_id="s1",
    )
    assert first_entry is not None
    first_created_at = first_entry.created_at
    entry_count_after_first = len(memory._entries)

    await agent._get_or_cache_chunks(chunks, session_id="s1", doc_hash="abc")

    assert len(memory._entries) == entry_count_after_first
    second_entry = memory.retrieve(
        key=f"doc_chunk:abc:{chunks[0].chunk_id}",
        scope=MemoryScope.SESSION,
        session_id="s1",
    )
    assert second_entry is not None
    assert second_entry.created_at == first_created_at


@pytest.mark.asyncio
async def test_cache_miss_stores_chunk() -> None:
    memory = InMemoryStore()
    agent = DocumentAgent(make_context(b"%PDF"), FakeProvider(), memory=memory)
    chunks = [make_chunk("hello world content for storage")]

    await agent._get_or_cache_chunks(chunks, session_id="s1", doc_hash="abc")

    entry = memory.retrieve(
        key=f"doc_chunk:abc:{chunks[0].chunk_id}",
        scope=MemoryScope.SESSION,
        session_id="s1",
    )
    assert entry is not None


@pytest.mark.asyncio
async def test_document_agent_works_without_memory() -> None:
    agent = DocumentAgent(make_context(b"%PDF"), FakeProvider(), memory=None)
    agent._extract_text = lambda content: sample_pages()  # type: ignore[method-assign]

    result = await agent.run()

    assert result.success is True


@pytest.mark.asyncio
async def test_empty_extraction_returns_failure() -> None:
    agent = DocumentAgent(make_context(b""), FakeProvider())
    agent._extract_text = lambda content: []  # type: ignore[method-assign]

    result = await agent.run()

    assert result.success is False
    assert result.error is not None
    assert "no text" in result.error.lower()


@pytest.mark.asyncio
async def test_evidence_fields_populated() -> None:
    agent = DocumentAgent(make_context(b"%PDF"), FakeProvider())
    agent._extract_text = lambda content: sample_pages()  # type: ignore[method-assign]

    result = await agent.run()

    assert len(result.evidence) > 0
    for evidence in result.evidence:
        assert isinstance(evidence, dict)
        assert "chunk_id" in evidence
        assert "section_path" in evidence
        assert "page_number" in evidence
        assert "excerpt" in evidence


@pytest.mark.asyncio
async def test_circuit_breaker_open_returns_failure() -> None:
    breaker = CircuitBreaker(name="llm", rolling_window=2, failure_threshold=0.5)
    breaker._transition(BreakerState.OPEN)
    breaker._opened_at = datetime.now(UTC)
    agent = DocumentAgent(
        make_context(b"%PDF"),
        FakeProvider(),
        circuit_breaker=breaker,
    )
    agent._extract_text = lambda content: sample_pages()  # type: ignore[method-assign]

    result = await agent.run()

    assert result.success is False
    assert result.error is not None
    assert "Circuit breaker" in result.error


@pytest.mark.asyncio
async def test_reasoning_steps_include_litm_note() -> None:
    agent = DocumentAgent(make_context(b"%PDF"), FakeProvider())
    agent._extract_text = lambda content: sample_pages()  # type: ignore[method-assign]

    result = await agent.run()

    litm_steps = [step for step in result.reasoning_steps if "Lost in the Middle" in step]
    assert len(litm_steps) == 1
