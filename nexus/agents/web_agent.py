from __future__ import annotations

from datetime import datetime
from typing import Any

from nexus.core.agent import AgentContext, AgentResult, BaseAgent, utc_now
from nexus.core.circuit_breaker import CircuitBreaker, CircuitOpenError
from nexus.memory.base import MemoryStore
from nexus.observability.schema import TraceEvent, TraceEventType
from nexus.observability.tracer import Tracer
from nexus.providers.base import LLMProvider
from nexus.tools import ToolPermission, ToolRegistry


class WebAgent(BaseAgent):
    """Agent that retrieves web evidence and synthesizes an answer from it."""

    def __init__(
        self,
        context: AgentContext,
        provider: LLMProvider,
        *,
        memory: MemoryStore | None = None,
        tool_registry: ToolRegistry | None = None,
        max_results: int = 5,
        tracer: Tracer | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        """Create a web agent with network tool permission."""
        super().__init__(
            context,
            provider,
            memory=memory,
            tool_registry=tool_registry,
            tracer=tracer,
        )
        self.max_results = max_results
        self._circuit_breaker = circuit_breaker
        self.agent_permissions = {ToolPermission.NETWORK}

    async def execute(self) -> AgentResult:
        """Retrieve web results, reason over them, and surface cited evidence."""
        if self._tool_registry is None:
            raise RuntimeError("tool registry is required for WebAgent")

        tool_name = "web_search"
        inputs: dict[str, Any] = {"query": self.context.task, "max_results": self.max_results}
        tracing = self.context.trace_id is not None and self.context.root_span_id is not None
        tool_span_id: str | None = None
        tool_start: datetime | None = None
        if tracing:
            assert self.context.trace_id is not None
            assert self.context.root_span_id is not None
            tool_span_id = self.tracer.start_span(
                trace_id=self.context.trace_id,
                agent_id=self.context.agent_id,
                parent_span_id=self.context.root_span_id,
            )
            self.tracer.record_event(
                tool_span_id,
                TraceEvent(
                    trace_id=self.context.trace_id,
                    span_id=tool_span_id,
                    parent_span_id=self.context.root_span_id,
                    event_type=TraceEventType.TOOL_CALL,
                    agent_id=self.context.agent_id,
                    timestamp=utc_now(),
                    payload={"tool_name": tool_name, "inputs": str(inputs)[:300]},
                ),
            )
            tool_start = utc_now()

        try:
            if self._circuit_breaker is not None:
                tool_result = await self._circuit_breaker.call(
                    self._tool_registry.execute,
                    tool_name,
                    inputs,
                    self.agent_permissions,
                )
            else:
                tool_result = await self._tool_registry.execute(
                    tool_name,
                    inputs,
                    self.agent_permissions,
                )
        except CircuitOpenError as exc:
            if tracing and tool_span_id is not None:
                assert self.context.trace_id is not None
                assert self.context.root_span_id is not None
                assert tool_start is not None
                tool_duration_ms = (utc_now() - tool_start).total_seconds() * 1000
                self.tracer.record_event(
                    tool_span_id,
                    TraceEvent(
                        trace_id=self.context.trace_id,
                        span_id=tool_span_id,
                        parent_span_id=self.context.root_span_id,
                        event_type=TraceEventType.TOOL_RESULT,
                        agent_id=self.context.agent_id,
                        timestamp=utc_now(),
                        duration_ms=tool_duration_ms,
                        payload={"success": False, "circuit_open": True},
                    ),
                )
                self.tracer.end_span(tool_span_id, "error", tool_duration_ms)
            breaker_name = exc.breaker_name
            return AgentResult(
                output="",
                success=False,
                error=f"Circuit breaker open: {tool_name}. Service degraded.",
                reasoning_steps=[
                    f"Circuit breaker '{breaker_name}' tripped; retry after "
                    f"{exc.retry_after.isoformat()}."
                ],
            )

        if tracing and tool_span_id is not None:
            assert self.context.trace_id is not None
            assert self.context.root_span_id is not None
            assert tool_start is not None
            tool_duration_ms = (utc_now() - tool_start).total_seconds() * 1000
            self.tracer.record_event(
                tool_span_id,
                TraceEvent(
                    trace_id=self.context.trace_id,
                    span_id=tool_span_id,
                    parent_span_id=self.context.root_span_id,
                    event_type=TraceEventType.TOOL_RESULT,
                    agent_id=self.context.agent_id,
                    timestamp=utc_now(),
                    duration_ms=tool_duration_ms,
                    payload={"success": tool_result.success},
                ),
            )
            self.tracer.end_span(
                tool_span_id,
                "complete" if tool_result.success else "error",
                tool_duration_ms,
            )
        if not tool_result.success:
            return AgentResult(
                output="",
                success=False,
                error=tool_result.error or "web_search failed",
            )

        results = self._normalize_results(tool_result.output)
        if not results:
            raise RuntimeError("web_search returned no results")

        system_prompt = (
            "You are a research agent. Answer only from the retrieved sources. "
            "Ground every claim in the provided titles, URLs, and snippets."
        )
        user_prompt = self._build_user_prompt(self.context.task, results)
        synthesis = await self.provider.complete_with_system(system_prompt, user_prompt)

        return AgentResult(
            output=synthesis,
            success=True,
            reasoning_steps=self._build_reasoning_steps(results),
            evidence=[result["url"] for result in results],
            metadata={"tool": "web_search", "result_count": len(results)},
        )

    @staticmethod
    def _normalize_results(output: Any) -> list[dict[str, str]]:
        """Convert tool output into title, URL, snippet dictionaries."""
        if not isinstance(output, list):
            raise RuntimeError("web_search returned an invalid result shape")

        normalized: list[dict[str, str]] = []
        for item in output:
            if not isinstance(item, dict):
                raise RuntimeError("web_search returned an invalid result shape")
            normalized.append(
                {
                    "title": str(item.get("title", "")),
                    "url": str(item.get("url", "")),
                    "snippet": str(item.get("snippet", "")),
                }
            )
        return normalized

    @staticmethod
    def _build_user_prompt(query: str, results: list[dict[str, str]]) -> str:
        """Build the grounded synthesis prompt from retrieved results."""
        sources = []
        for index, result in enumerate(results, start=1):
            sources.append(
                "\n".join(
                    [
                        f"Source {index}: {result['title']}",
                        f"URL: {result['url']}",
                        f"Snippet: {result['snippet']}",
                    ]
                )
            )
        return f"Question: {query}\n\nRetrieved sources:\n\n" + "\n\n".join(sources)

    @staticmethod
    def _build_reasoning_steps(results: list[dict[str, str]]) -> list[str]:
        """Create a deterministic reasoning audit trail from retrieved evidence."""
        return [
            f"Retrieved {result['title']} from {result['url']}: {result['snippet']}"
            for result in results
        ]
