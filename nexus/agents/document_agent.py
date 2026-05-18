from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from io import BytesIO
from typing import Any

import structlog
from pypdf import PdfReader

from nexus.core.agent import AgentContext, AgentResult, BaseAgent, utc_now
from nexus.core.circuit_breaker import CircuitBreaker, CircuitOpenError
from nexus.memory.base import MemoryEntry, MemoryScope, MemoryStore
from nexus.observability.tracer import Tracer
from nexus.providers.base import LLMProvider

_logger = structlog.get_logger()


@dataclass
class DocumentChunk:
    chunk_id: str
    content: str
    section_path: list[str]
    page_number: int
    position_in_section: int
    token_count: int
    content_hash: str


@dataclass
class ChunkingConfig:
    max_tokens_per_chunk: int = 512
    overlap_tokens: int = 64
    min_tokens_per_chunk: int = 64
    respect_sentence_boundaries: bool = True
    respect_paragraph_boundaries: bool = True


class DocumentAgent(BaseAgent):
    """Agent that ingests documents, chunks semantically, and answers with citations."""

    def __init__(
        self,
        context: AgentContext,
        provider: LLMProvider,
        *,
        memory: MemoryStore | None = None,
        tracer: Tracer | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        default_chunking_config: ChunkingConfig | None = None,
    ) -> None:
        super().__init__(context, provider, memory=memory, tracer=tracer)
        self._circuit_breaker = circuit_breaker
        self.chunking_config = default_chunking_config or ChunkingConfig()
        self._last_cache_hits = 0
        self._last_cache_misses = 0

    def _extract_text(self, content: bytes) -> list[tuple[int, str, list[str]]]:
        """Return (page_number, text, section_path) tuples extracted from PDF bytes."""
        pages: list[tuple[int, str, list[str]]] = []
        section_path: list[str] = []

        try:
            reader = PdfReader(BytesIO(content))
        except Exception as exc:
            _logger.warning("pdf_read_failed", error=str(exc))
            return pages

        for page_index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                _logger.warning("pdf_page_empty", page_number=page_index)
                continue

            page_sections: list[tuple[int, str, list[str]]] = []
            lines = text.splitlines()
            buffer: list[str] = []
            current_path = list(section_path)

            for line_idx, line in enumerate(lines):
                next_line = lines[line_idx + 1] if line_idx + 1 < len(lines) else None
                if self._is_section_header(line, next_line):
                    if buffer:
                        joined = "\n".join(buffer).strip()
                        if joined:
                            page_sections.append((page_index, joined, list(current_path)))
                        buffer.clear()
                    header = line.strip()
                    current_path = [header]
                    section_path = list(current_path)
                    continue
                buffer.append(line)

            if buffer:
                joined = "\n".join(buffer).strip()
                if joined:
                    page_sections.append((page_index, joined, list(current_path)))
            pages.extend(page_sections)

        return pages

    @staticmethod
    def _is_section_header(line: str, next_line: str | None) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if stripped.isupper() and any(char.isalpha() for char in stripped):
            return True
        return (
            next_line is not None
            and not next_line.strip()
            and len(stripped) < 80
        )

    def _chunk_document(
        self,
        pages: list[tuple[int, str, list[str]]],
        config: ChunkingConfig,
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        overlap_tail = ""

        for page_number, text, section_path in pages:
            paragraphs = (
                re.split(r"\n\n+", text) if config.respect_paragraph_boundaries else [text]
            )
            position = 0

            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue

                units = [paragraph]
                over_cap = self._token_count(paragraph) > config.max_tokens_per_chunk
                if config.respect_sentence_boundaries and over_cap:
                    units = self._split_sentences(paragraph)

                for unit in units:
                    pieces = self._enforce_token_cap(unit, config.max_tokens_per_chunk)
                    for piece in pieces:
                        content = piece
                        if overlap_tail:
                            content = f"{overlap_tail} {piece}".strip()
                        token_count = self._token_count(content)
                        if token_count < config.min_tokens_per_chunk:
                            overlap_tail = self._overlap_suffix(content, config.overlap_tokens)
                            continue

                        chunk = self._make_chunk(
                            content=content,
                            section_path=section_path,
                            page_number=page_number,
                            position_in_section=position,
                        )
                        chunks.append(chunk)
                        position += 1
                        overlap_tail = self._overlap_suffix(content, config.overlap_tokens)

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        parts = re.split(r"(?<=\. )\s+", text)
        return [part.strip() for part in parts if part.strip()]

    def _enforce_token_cap(self, text: str, max_tokens: int) -> list[str]:
        if self._token_count(text) <= max_tokens:
            return [text]

        _logger.warning("chunk_hard_trim", original_tokens=self._token_count(text), cap=max_tokens)
        words = text.split()
        return [" ".join(words[:max_tokens])]

    @staticmethod
    def _token_count(text: str) -> int:
        return len(text.split())

    @staticmethod
    def _overlap_suffix(text: str, overlap_tokens: int) -> str:
        words = text.split()
        if overlap_tokens <= 0 or not words:
            return ""
        return " ".join(words[-overlap_tokens:])

    @staticmethod
    def _make_chunk(
        *,
        content: str,
        section_path: list[str],
        page_number: int,
        position_in_section: int,
    ) -> DocumentChunk:
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        return DocumentChunk(
            chunk_id=content_hash[:16],
            content=content,
            section_path=section_path,
            page_number=page_number,
            position_in_section=position_in_section,
            token_count=len(content.split()),
            content_hash=content_hash,
        )

    def _assemble_context(
        self,
        chunks: list[DocumentChunk],
        scored_chunks: list[tuple[float, DocumentChunk]],
        max_context_tokens: int = 4000,
    ) -> list[DocumentChunk]:
        if not scored_chunks:
            return []

        ordered = self._lost_in_middle_order(scored_chunks)
        assembled: list[DocumentChunk] = []
        total_tokens = 0

        for index, chunk in enumerate(ordered):
            must_include = index == 0 or index == len(ordered) - 1
            if not must_include and total_tokens + chunk.token_count > max_context_tokens:
                continue
            assembled.append(chunk)
            total_tokens += chunk.token_count

        return assembled

    @staticmethod
    def _lost_in_middle_order(
        scored_chunks: list[tuple[float, DocumentChunk]],
    ) -> list[DocumentChunk]:
        if not scored_chunks:
            return []
        if len(scored_chunks) == 1:
            return [scored_chunks[0][1]]
        best = scored_chunks[0][1]
        second = scored_chunks[1][1]
        middle = [chunk for _, chunk in scored_chunks[2:]]
        return [best, *middle, second]

    def _score_chunks(
        self,
        chunks: list[DocumentChunk],
        query: str,
    ) -> list[tuple[float, DocumentChunk]]:
        query_terms = set(query.lower().split())
        scored: list[tuple[float, DocumentChunk]] = []
        for chunk in chunks:
            chunk_terms = set(chunk.content.lower().split())
            overlap = len(query_terms & chunk_terms)
            score = overlap / (len(query_terms) + 1e-6)
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    async def _get_or_cache_chunks(
        self,
        chunks: list[DocumentChunk],
        session_id: str,
        doc_hash: str,
    ) -> list[DocumentChunk]:
        if self.memory is None:
            self._last_cache_hits = 0
            self._last_cache_misses = len(chunks)
            return chunks

        hits = 0
        misses = 0
        resolved: list[DocumentChunk] = []

        for chunk in chunks:
            cache_key = f"doc_chunk:{doc_hash}:{chunk.chunk_id}"
            entry = self.memory.retrieve(
                key=cache_key,
                scope=MemoryScope.SESSION,
                session_id=session_id,
            )
            if entry is not None:
                hits += 1
                resolved.append(self._deserialize_chunk(entry.content))
                continue

            misses += 1
            payload = self._serialize_chunk(chunk, doc_hash)
            self.memory.store(
                cache_key,
                MemoryEntry(
                    content=payload,
                    scope=MemoryScope.SESSION,
                    agent_id="document_agent",
                    session_id=session_id,
                    created_at=utc_now(),
                    ttl_seconds=3600,
                ),
            )
            resolved.append(chunk)

        self._last_cache_hits = hits
        self._last_cache_misses = misses
        return resolved

    @staticmethod
    def _serialize_chunk(chunk: DocumentChunk, source_doc_hash: str) -> str:
        data = asdict(chunk)
        data["source_doc_hash"] = source_doc_hash
        return json.dumps(data)

    @staticmethod
    def _deserialize_chunk(payload: str) -> DocumentChunk:
        data = json.loads(payload)
        return DocumentChunk(
            chunk_id=data["chunk_id"],
            content=data["content"],
            section_path=data["section_path"],
            page_number=data["page_number"],
            position_in_section=data["position_in_section"],
            token_count=data["token_count"],
            content_hash=data["content_hash"],
        )

    def _parse_chunking_config(self, overrides: dict[str, Any]) -> ChunkingConfig:
        base = asdict(self.chunking_config)
        base.update(overrides)
        return ChunkingConfig(**base)

    def _parse_run_input(self) -> tuple[bytes, str, str, ChunkingConfig]:
        config = self.context.config
        content = config.get("content")
        query = config.get("query", self.context.task)
        session_id = config.get("session_id", "default-session")
        chunking_overrides = config.get("chunking_config") or {}

        if not isinstance(content, (bytes, bytearray)):
            raise ValueError("context.config['content'] must be PDF bytes")
        if not isinstance(query, str):
            raise ValueError("context.config['query'] must be a string")
        if not isinstance(session_id, str):
            raise ValueError("context.config['session_id'] must be a string")

        return (
            bytes(content),
            query,
            session_id,
            self._parse_chunking_config(chunking_overrides),
        )

    def _format_context_block(self, chunks: list[DocumentChunk]) -> str:
        blocks: list[str] = []
        for chunk in chunks:
            section = " > ".join(chunk.section_path) if chunk.section_path else "Document"
            blocks.append(
                "\n".join(
                    [
                        f"[Section: {section} | Page: {chunk.page_number}]",
                        chunk.content,
                    ]
                )
            )
        return "\n\n".join(blocks)

    async def execute(self) -> AgentResult:
        content, query, session_id, chunking_config = self._parse_run_input()
        pages = self._extract_text(content)

        if not pages:
            return AgentResult(
                output="",
                success=False,
                error="Document extraction produced no text",
                reasoning_steps=["Document extraction produced no pages with text"],
                evidence=[],
            )

        chunks = self._chunk_document(pages, chunking_config)
        doc_hash = hashlib.sha256(content).hexdigest()[:16]
        chunks = await self._get_or_cache_chunks(chunks, session_id, doc_hash)
        scored_chunks = self._score_chunks(chunks, query)
        assembled = self._assemble_context(chunks, scored_chunks)
        total_tokens = sum(chunk.token_count for chunk in assembled)
        score_map = {chunk.chunk_id: score for score, chunk in scored_chunks}

        system_prompt = (
            "You are answering a question based only on the provided document excerpts. "
            "Cite the section path and page number for each claim."
        )
        user_prompt = "\n".join(
            [
                "[DOCUMENT EXCERPTS]",
                self._format_context_block(assembled),
                "",
                "[QUESTION]",
                query,
            ]
        )

        try:
            if self._circuit_breaker is not None:
                llm_response = await self._circuit_breaker.call(
                    self.provider.complete_with_system,
                    system_prompt,
                    user_prompt,
                )
            else:
                llm_response = await self.provider.complete_with_system(
                    system_prompt,
                    user_prompt,
                )
        except CircuitOpenError:
            return AgentResult(
                output="",
                success=False,
                error="Circuit breaker open: LLM provider unavailable",
                reasoning_steps=["Circuit breaker tripped — skipping LLM call"],
                evidence=[],
            )

        evidence = [
            {
                "chunk_id": chunk.chunk_id,
                "section_path": chunk.section_path,
                "page_number": chunk.page_number,
                "relevance_score": score_map.get(chunk.chunk_id, 0.0),
                "excerpt": chunk.content[:200],
            }
            for chunk in assembled
        ]

        reasoning_steps = [
            f"Extracted {len(pages)} pages, produced {len(chunks)} chunks",
            f"Cache: {self._last_cache_hits} hits, {self._last_cache_misses} misses",
            f"Assembled {len(assembled)} chunks into context ({total_tokens} tokens)",
            (
                "Applied Lost in the Middle ordering: top chunk at position 0, "
                "second-top at position -1"
            ),
        ]

        return AgentResult(
            output=llm_response,
            success=True,
            reasoning_steps=reasoning_steps,
            evidence=evidence,
        )
