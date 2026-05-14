from __future__ import annotations

from typing import Protocol, runtime_checkable

from .schema import TraceEvent, TraceEventType, TraceSpan
from .tracer import InMemoryTracer


@runtime_checkable
class TraceStore(Protocol):
    def get_trace(self, trace_id: str) -> list[TraceSpan]:
        """Return all spans for a given trace_id, ordered by start time."""
        ...

    def get_recent(self, agent_id: str, n: int) -> list[TraceSpan]:
        """Return the n most recent root spans for an agent."""
        ...

    def get_events_by_type(
        self, trace_id: str, event_type: TraceEventType
    ) -> list[TraceEvent]:
        """Return all events of a given type across all spans in a trace."""
        ...


class InMemoryTraceStore:
    """
    Read interface over InMemoryTracer's internal data.
    Shares the same tracer instance — reads from the same dict the tracer writes to.
    """

    def __init__(self, tracer: InMemoryTracer) -> None:
        self._tracer = tracer

    def get_trace(self, trace_id: str) -> list[TraceSpan]:
        spans = self._tracer.get_spans_for_trace(trace_id)
        return sorted(spans, key=lambda s: s.started_at)

    def get_recent(self, agent_id: str, n: int) -> list[TraceSpan]:
        """Returns root spans (no parent) for the agent, most recent first."""
        all_spans = self._tracer.get_all_spans_for_agent(agent_id)
        root_spans = [s for s in all_spans if s.parent_span_id is None]
        return sorted(root_spans, key=lambda s: s.started_at, reverse=True)[:n]

    def get_events_by_type(
        self, trace_id: str, event_type: TraceEventType
    ) -> list[TraceEvent]:
        spans = self.get_trace(trace_id)
        return [
            event
            for span in spans
            for event in span.events
            if event.event_type == event_type
        ]
