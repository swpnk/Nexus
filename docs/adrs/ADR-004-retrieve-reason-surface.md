# ADR-004: Agents Follow Retrieve, Reason, Surface

## Status

Accepted

## Date

2026-05-13

## Context

Day 3 introduces `WebAgent`, the first Nexus agent that answers a user question. The previous layers established lifecycle, memory, and permission-gated tools, but they did not define how an agent should produce a verifiable answer.

Production agents need more than output text. They need a repeatable pattern for collecting source material, reasoning over it, and returning an audit trail that downstream consumers can inspect.

## Decision

Concrete agents that rely on external information follow the retrieve, reason, surface pattern:

- Retrieve: call tools through `ToolRegistry`, letting the registry enforce permissions before tool execution.
- Reason: pass retrieved material to the LLM provider in a structured prompt that asks the model to synthesize from sources rather than from memory.
- Surface: return the synthesis in `AgentResult.output`, reasoning in `AgentResult.reasoning_steps`, and source identifiers in `AgentResult.evidence`.

`WebAgent` is the first implementation of this pattern.

## Consequences

- Tool permission boundaries remain centralized in `ToolRegistry`.
- Agent outputs become inspectable because evidence is returned beside the answer.
- Future tracing and evaluation systems can use `reasoning_steps` and `evidence` without parsing free-form output.
- Agents that answer from external information but do not surface evidence are architectural debt.

## Failure Mode

The dangerous failure is not a failed tool call. It is a successful retrieval followed by a synthesis that ignores the retrieved sources and answers from model priors.

This can look correct because the answer is fluent and the evidence field is populated. Code can enforce that retrieval happened and evidence was surfaced, but it cannot prove that every generated claim is grounded. That requires prompt discipline now and evidence-grounding evaluations later.

## Alternatives Considered

- Return text only: rejected because consumers cannot audit or verify the answer.
- Let agents call tools directly: rejected because it bypasses permission enforcement and weakens the control-plane/data-plane split.
- Store citations only in the output text: rejected because citations become hard to inspect programmatically and easy to omit.

## Known Limitations

- `WebAgent` uses deterministic stub search results in Day 3. Real search arrives later.
- `AgentResult.evidence` currently stores source URLs as strings. Rich source metadata can be added once the evidence contract is widened deliberately.
- Grounding quality is not automatically evaluated yet. Week 4 evaluations should check whether output claims are consistent with surfaced evidence.
