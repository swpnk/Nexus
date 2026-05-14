# ADR-004: Observability Tracer Design

## Status

Accepted

## Context

Every agent run must be auditable retroactively. Logs are insufficient — they have no causal model, no parent/child relationships, and no queryable store. The system needs structured traces.

## Decision

Implement a Protocol-based `Tracer` interface with `InMemoryTracer` as Day 4's backend.

## Consequences

- `InMemoryTracer` is zero-dependency, zero-latency, fully testable
- Protocol interface means OpenTelemetry exporter is a one-file swap (Day 28)
- `record_event()` swallows its own exceptions — tracer failures degrade the trace, never the agent
- `trace_id` propagates through `AgentContext` — foundation for Week 3 mandatory cross-agent trace propagation

## Rejected alternative

Wrapping Python's stdlib `logging` module. Rejected because: no span model, no parent/child, no queryable store. Would require building the span model on top of logging anyway.

## What this defers

- Persistence across restarts
- Distributed trace context propagation (OpenTelemetry W3C TraceContext headers)
- Sampling policy
- Export to Jaeger / Tempo / Honeycomb

These are Week 4 concerns.
