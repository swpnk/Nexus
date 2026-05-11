# ADR-001: Base Architecture

## Status

Accepted

## Context

Nexus Day 0 establishes the foundational contract that future agents build on. The goal is a small, strict architecture that makes lifecycle, failure handling, state transitions, configuration, and provider selection predictable from the beginning.

## Decisions

### 1. `run()` is final

**Decision:** `BaseAgent.run()` is marked final, and subclasses must not override it.

**Rationale:** The lifecycle must remain consistent for every agent. A final `run()` guarantees transition validation, duration measurement, error conversion, and cleanup are always applied.

**Rejected alternative:** Letting subclasses override `run()` would make each agent responsible for the lifecycle and would allow resource leaks, missing transitions, or exceptions escaping inconsistently.

### 2. `execute()` is the only subclass extension point

**Decision:** Subclasses implement `execute()` for agent-specific work.

**Rationale:** Keeping one extension point makes agent behavior easy to reason about while preserving the framework-owned lifecycle around it.

**Rejected alternative:** Exposing several lifecycle hooks would increase ordering complexity before the framework has a concrete need for them.

### 3. `cleanup()` is guaranteed via `finally`

**Decision:** `run()` always calls `cleanup()` in a `finally` block.

**Rationale:** Agents may later own network clients, temporary files, or other resources. Cleanup must run whether execution succeeds or fails.

**Rejected alternative:** Asking subclasses or callers to invoke cleanup manually would make resource management optional and error-prone.

### 4. `AgentResult` is returned, never raised

**Decision:** Agent failures are converted into `AgentResult(success=False)` rather than re-raised from `run()`.

**Rationale:** Callers get one stable result contract for success and failure, which simplifies orchestration and downstream reporting.

**Rejected alternative:** Raising exceptions from `run()` would force every caller to duplicate failure handling and would fragment agent outcomes across return values and exceptions.

### 5. `VALID_TRANSITIONS` is a data structure

**Decision:** Valid state changes live in the `VALID_TRANSITIONS` dictionary.

**Rationale:** State machine rules should be inspectable, testable, and easy to extend without embedding policy in branching logic.

**Rejected alternative:** Encoding transitions with `if`/`else` blocks would make the rules harder to audit and easier to accidentally diverge from the documented contract.

### 6. LLM providers are injected via factory

**Decision:** Agents receive an `LLMProvider` from the provider factory and never import provider SDKs directly.

**Rationale:** Dependency injection keeps agents decoupled from vendor SDKs, makes testing straightforward, and preserves a stable provider protocol.

**Rejected alternative:** Importing Anthropic or OpenAI clients inside agent code would tightly couple agent implementations to vendors and make mocked tests harder.
