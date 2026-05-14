from __future__ import annotations

from datetime import UTC, datetime

import pytest

from nexus.core.agent import AgentContext, AgentResult, BaseAgent
from pydantic import ValidationError

from nexus.observability.schema import TraceEvent, TraceEventType, TraceSpan
from nexus.observability.store import InMemoryTraceStore
from nexus.observability.tracer import InMemoryTracer
from nexus.providers.base import LLMProvider


class FakeProvider:
    @property
    def model(self) -> str:
        return "fake-model"

    async def complete(self, prompt: str, **kwargs: object) -> str:
        return f"completed: {prompt}"

    async def complete_with_system(self, system: str, user: str, **kwargs: object) -> str:
        return f"{system}: {user}"


class SuccessfulTraceAgent(BaseAgent):
    def __init__(
        self,
        context: AgentContext,
        provider: LLMProvider,
        tracer: InMemoryTracer,
    ) -> None:
        super().__init__(context, provider, tracer=tracer)

    async def execute(self) -> AgentResult:
        return AgentResult(output="ok", success=True)


class FailingTraceAgent(BaseAgent):
    def __init__(
        self,
        context: AgentContext,
        provider: LLMProvider,
        tracer: InMemoryTracer,
    ) -> None:
        super().__init__(context, provider, tracer=tracer)

    async def execute(self) -> AgentResult:
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_trace_emitted_on_success() -> None:
    """Running a successful agent emits AGENT_START and AGENT_COMPLETE events."""
    tracer = InMemoryTracer()
    context = AgentContext(agent_id="agent-1", task="hello")
    agent = SuccessfulTraceAgent(context, FakeProvider(), tracer)

    await agent.run()

    assert agent.context.trace_id is not None
    assert agent.context.root_span_id is not None
    store = InMemoryTraceStore(tracer)
    event_types = {
        e.event_type
        for s in store.get_trace(agent.context.trace_id)
        for e in s.events
    }
    assert TraceEventType.AGENT_START in event_types
    assert TraceEventType.AGENT_COMPLETE in event_types
    trace = store.get_trace(agent.context.trace_id)
    root = next(s for s in trace if s.span_id == agent.context.root_span_id)
    assert root.status == "complete"


@pytest.mark.asyncio
async def test_trace_emitted_on_failure() -> None:
    """An agent that raises emits AGENT_START and AGENT_ERROR events, span status is 'error'."""
    tracer = InMemoryTracer()
    context = AgentContext(agent_id="agent-1", task="hello")
    agent = FailingTraceAgent(context, FakeProvider(), tracer)

    await agent.run()

    assert agent.context.trace_id is not None
    store = InMemoryTraceStore(tracer)
    event_types = {
        e.event_type
        for s in store.get_trace(agent.context.trace_id)
        for e in s.events
    }
    assert TraceEventType.AGENT_START in event_types
    assert TraceEventType.AGENT_ERROR in event_types
    trace = store.get_trace(agent.context.trace_id)
    root = next(s for s in trace if s.span_id == agent.context.root_span_id)
    assert root.status == "error"


def test_tracer_exception_does_not_propagate() -> None:
    """If record_event internally raises, the exception must NOT reach the caller."""
    tracer = InMemoryTracer()
    event = TraceEvent(
        trace_id="test-trace",
        span_id="nonexistent-span",
        event_type=TraceEventType.AGENT_START,
        agent_id="test-agent",
        timestamp=datetime.now(UTC),
    )
    tracer.record_event("nonexistent-span", event)


def test_child_spans_have_correct_parent() -> None:
    """Tool call spans have parent_span_id equal to the root span_id."""
    tracer = InMemoryTracer()
    trace_id = "trace-001"
    root_span_id = tracer.start_span(trace_id, "agent-001")
    child_span_id = tracer.start_span(trace_id, "agent-001", parent_span_id=root_span_id)

    store = InMemoryTraceStore(tracer)
    spans = store.get_trace(trace_id)

    child = next(s for s in spans if s.span_id == child_span_id)
    assert child.parent_span_id == root_span_id


def test_all_datetimes_are_utc_aware() -> None:
    """TraceEvent and TraceSpan must reject naive datetimes."""
    with pytest.raises(ValidationError):
        TraceEvent(
            trace_id="t",
            span_id="s",
            event_type=TraceEventType.AGENT_START,
            agent_id="a",
            timestamp=datetime(2024, 1, 1),
        )

    with pytest.raises(ValidationError):
        TraceSpan(
            trace_id="t",
            span_id="s",
            agent_id="a",
            started_at=datetime(2024, 1, 1),
        )


def test_store_get_trace_returns_full_tree() -> None:
    """get_trace returns all spans for the trace, ordered by start time."""
    tracer = InMemoryTracer()
    trace_id = "trace-full"
    root_span_id = tracer.start_span(trace_id, "agent-001")
    child_span_id = tracer.start_span(trace_id, "agent-001", parent_span_id=root_span_id)

    store = InMemoryTraceStore(tracer)
    spans = store.get_trace(trace_id)

    assert len(spans) == 2
    span_ids = {s.span_id for s in spans}
    assert root_span_id in span_ids
    assert child_span_id in span_ids
