from __future__ import annotations

import sys
import threading
from datetime import UTC, datetime
from typing import Literal, Protocol, runtime_checkable

from .schema import TraceEvent, TraceSpan, generate_span_id


@runtime_checkable
class Tracer(Protocol):
    def start_span(
        self,
        trace_id: str,
        agent_id: str,
        parent_span_id: str | None = None,
    ) -> str:
        """Open a new span. Returns span_id."""
        ...

    def record_event(self, span_id: str, event: TraceEvent) -> None:
        """Append an event to an open span. Must never raise."""
        ...

    def end_span(
        self,
        span_id: str,
        status: Literal["complete", "error"],
        duration_ms: float,
    ) -> None:
        """Close a span with final status and duration."""
        ...


class NoOpTracer:
    """Use in tests that don't care about observability."""

    def start_span(
        self,
        trace_id: str,
        agent_id: str,
        parent_span_id: str | None = None,
    ) -> str:
        return generate_span_id()

    def record_event(self, span_id: str, event: TraceEvent) -> None:
        pass

    def end_span(
        self,
        span_id: str,
        status: Literal["complete", "error"],
        duration_ms: float,
    ) -> None:
        pass


class InMemoryTracer:
    """
    Thread-safe in-memory tracer.

    Internal structure:
        _spans: dict[span_id, TraceSpan]
        _trace_index: dict[trace_id, list[span_id]]

    record_event() swallows its own exceptions — tracer failures must
    never propagate to the agent. Degraded trace > crashed agent.
    """

    def __init__(self) -> None:
        self._spans: dict[str, TraceSpan] = {}
        self._trace_index: dict[str, list[str]] = {}
        self._lock = threading.Lock()

    def start_span(
        self,
        trace_id: str,
        agent_id: str,
        parent_span_id: str | None = None,
    ) -> str:
        span_id = generate_span_id()
        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            agent_id=agent_id,
            started_at=datetime.now(UTC),
            status="running",
        )
        with self._lock:
            self._spans[span_id] = span
            if trace_id not in self._trace_index:
                self._trace_index[trace_id] = []
            self._trace_index[trace_id].append(span_id)
        return span_id

    def record_event(self, span_id: str, event: TraceEvent) -> None:
        try:
            with self._lock:
                span = self._spans.get(span_id)
                if span is None:
                    print(
                        f"[tracer] record_event: span {span_id} not found",
                        file=sys.stderr,
                    )
                    return
                span.events.append(event)
        except Exception as e:
            print(f"[tracer] record_event failed: {e}", file=sys.stderr)

    def end_span(
        self,
        span_id: str,
        status: Literal["complete", "error"],
        duration_ms: float,
    ) -> None:
        try:
            with self._lock:
                span = self._spans.get(span_id)
                if span is None:
                    return
                span.ended_at = datetime.now(UTC)
                span.status = status
        except Exception as e:
            print(f"[tracer] end_span failed: {e}", file=sys.stderr)

    def get_spans_for_trace(self, trace_id: str) -> list[TraceSpan]:
        with self._lock:
            span_ids = self._trace_index.get(trace_id, [])
            return [self._spans[sid] for sid in span_ids if sid in self._spans]

    def get_all_spans_for_agent(self, agent_id: str) -> list[TraceSpan]:
        with self._lock:
            return [s for s in self._spans.values() if s.agent_id == agent_id]
