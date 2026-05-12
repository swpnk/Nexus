from __future__ import annotations

from time import monotonic
from typing import Any

from nexus.tools.base import (
    ToolCallable,
    ToolDefinition,
    ToolNotFoundError,
    ToolPermission,
    ToolPermissionError,
    ToolResult,
)


class ToolRegistry:
    """Registry that resolves tools and enforces permission boundaries."""

    def __init__(self) -> None:
        """Create an empty tool registry."""
        self._tools: dict[str, ToolDefinition] = {}
        self._callables: dict[str, ToolCallable] = {}

    def register(self, definition: ToolDefinition, tool_callable: ToolCallable) -> None:
        """Register a tool definition and callable."""
        if definition.name in self._tools:
            raise ValueError(
                f"Tool '{definition.name}' is already registered. "
                "Use a different name or increment the version."
            )
        self._tools[definition.name] = definition
        self._callables[definition.name] = tool_callable

    def resolve(self, tool_name: str) -> ToolDefinition:
        """Return the ToolDefinition for a given name."""
        if tool_name not in self._tools:
            raise ToolNotFoundError(tool_name)
        return self._tools[tool_name]

    async def execute(
        self,
        tool_name: str,
        inputs: dict[str, Any],
        agent_permissions: set[ToolPermission],
    ) -> ToolResult:
        """Execute a tool after resolving it and enforcing permissions."""
        definition = self.resolve(tool_name)
        if definition.permission not in agent_permissions:
            raise ToolPermissionError(tool_name, definition.permission, agent_permissions)

        start = monotonic()
        try:
            output = await self._callables[tool_name](inputs)
            return ToolResult(
                success=True,
                output=output,
                execution_time_ms=self._duration_ms_since(start),
                tool_name=tool_name,
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                error=str(exc),
                execution_time_ms=self._duration_ms_since(start),
                tool_name=tool_name,
            )

    def list_by_permission(self, permission: ToolPermission) -> list[ToolDefinition]:
        """Return all tools whose permission exactly matches permission."""
        return [
            definition
            for definition in self._tools.values()
            if definition.permission == permission
        ]

    def list_all(self) -> list[ToolDefinition]:
        """Return all registered tool definitions."""
        return list(self._tools.values())

    def tool_descriptions_for_prompt(self) -> str:
        """Return tool descriptions formatted for LLM prompt injection."""
        lines = [
            f"- {definition.name}: {definition.description} "
            f"[permission: {definition.permission.value}]"
            for definition in self._tools.values()
        ]
        return "\n".join(lines)

    @staticmethod
    def _duration_ms_since(start: float) -> float:
        """Return elapsed milliseconds since a monotonic start value."""
        return (monotonic() - start) * 1000
