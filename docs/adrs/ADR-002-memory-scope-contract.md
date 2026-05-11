# ADR-002: Memory Scope as a Mandatory First-Class Contract

## Status

Accepted

## Context

Agent memory needs trust boundaries from Day 1 because stored data can outlive a single method call and may influence future decisions. If scope is optional or added later, early unscoped writes become ambiguous data that must be migrated, audited, and partitioned after the system already depends on it. Nexus treats memory isolation as part of the storage contract rather than a policy layered on after writes occur.

## Decision

Scope is mandatory on every `MemoryEntry` write. There is no default scope.

`SESSION` scope is the only fully implemented tier in Day 1. `USER`, `AGENT`, and `GLOBAL` stubs raise `NotImplementedError` pointing to Day 6.

## Consequences

- Scope errors are caught at write time, not after data has already been stored in the wrong bucket.
- Trust violations between sessions are architecturally impossible in `InMemoryStore`.
- Migration cost is zero because there is no existing unscoped data to partition.

## Alternatives Considered

- Global bucket with future migration: rejected because migration cost grows with data.
- Optional scope with `SESSION` default: rejected because it allows silent wrong-scope writes.

## Known Limitations

- `InMemoryStore` has no eviction policy. Working set grows unbounded.
  Fix: LRU plus TTL-based eviction in Day 6.
- Semantic search is not implemented. `search()` is recency-ordered exact match.
  Fix: vector backend in Day 6.
