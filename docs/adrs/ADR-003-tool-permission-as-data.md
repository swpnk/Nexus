# ADR-003: Tool Permission Is a Field on ToolDefinition, Not a Runtime Decorator

## Status

Accepted

## Date

2026-05-12

## Context

Agents need to invoke tools. Tools have different trust levels: reading memory is safer than filesystem access, network calls, or executing code. Nexus needs to enforce these boundaries at the architecture level, not through convention.

## Decision

`ToolPermission` is a required field on `ToolDefinition`. It is a data structure, not a decorator or naming convention. `ToolRegistry.execute()` checks the agent's declared permissions against the tool's required permission before invoking the callable.

## Consequences

- Permission is auditable: any code can inspect `tool_definition.permission` at any time.
- Permission is enforced before execution: the tool callable is never invoked without authorization.
- The orchestrator can construct per-agent registry views by querying permission fields.
- New tools must declare their permission at registration time because omission is a validation error.

## Alternatives Considered

- Decorator pattern (`@requires_permission(NETWORK)`): rejected because decorators are invisible to the registry. They cannot be inspected programmatically by the orchestrator without coupling to callable internals.
- Naming convention (`network__web_search`): rejected because conventions break silently. A data field is enforced by the model contract.

## Known Limitations

- Permission levels are coarse-grained. Fine-grained resource control, such as `NETWORK` access limited to specific domains, is not implemented.
  Fix: add `allowed_domains: list[str]` or a richer policy object to `ToolDefinition` in a future ADR.
