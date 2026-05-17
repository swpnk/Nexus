# ADR-005: Circuit Breaker as Generic Callable Wrapper

**Status:** Accepted
**Date:** Day 5

## Decision

CircuitBreaker wraps async callables generically. It is not aware of tools,
LLMs, or HTTP. Tool registry and LLM providers each receive their own injected
instance.

## Rationale

- Keeps the breaker testable in isolation
- Allows per-dependency threshold configuration (LLM failures tolerate higher
  rate than external APIs)
- Consistent with Day 0 principle: dependencies injected, never imported

## Tradeoff

Callers are responsible for injection. A default registry (CircuitBreakerRegistry)
is provided for convenience but not required.

## Consequences

Week 3 orchestrator can query the registry snapshot to make routing decisions
based on breaker state. Day 30 SLO tracker uses trip events from the tracer.
