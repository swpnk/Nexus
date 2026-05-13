from __future__ import annotations

from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict


class ToolPermission(StrEnum):
    """Permission tier required to execute a tool."""

    READ_ONLY = "read_only"
    NETWORK = "network"
    FILESYSTEM = "filesystem"
    EXECUTE = "execute"


class ToolDefinition(BaseModel):
    """Immutable metadata contract for a registered tool."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    permission: ToolPermission
    version: str = "1.0.0"


class ToolResult(BaseModel):
    """Normalized outcome returned by tool execution."""

    success: bool
    output: Any = None
    error: str | None = None
    execution_time_ms: float
    tool_name: str


class ToolError(Exception):
    """Base exception for all tool-related errors."""


class ToolNotFoundError(ToolError):
    """Raised when a tool name cannot be resolved in the registry."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' not found in registry")


class ToolPermissionError(ToolError):
    """Raised when an agent attempts to invoke a tool without permission."""

    def __init__(
        self,
        tool_name: str,
        required: ToolPermission,
        agent_permissions: set[ToolPermission],
    ) -> None:
        self.tool_name = tool_name
        self.required = required
        self.agent_permissions = agent_permissions
        permissions = [permission.value for permission in agent_permissions]
        super().__init__(
            f"Tool '{tool_name}' requires permission '{required.value}' "
            f"but agent only has: {permissions}"
        )


class ToolExecutionError(ToolError):
    """Raised when a tool callable raises an unhandled exception."""

    def __init__(self, tool_name: str, cause: Exception) -> None:
        self.tool_name = tool_name
        self.cause = cause
        super().__init__(f"Tool '{tool_name}' raised during execution: {cause}")


ToolCallable = Callable[[dict[str, Any]], Awaitable[Any]]
