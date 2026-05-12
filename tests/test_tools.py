from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from nexus.core.agent import AgentContext, AgentResult, BaseAgent
from nexus.providers.base import LLMProvider
from nexus.tools import (
    ToolDefinition,
    ToolNotFoundError,
    ToolPermission,
    ToolPermissionError,
    ToolRegistry,
    ToolResult,
)
from nexus.tools.builtins.web_search import WEB_SEARCH_DEFINITION, web_search_callable


class FakeProvider:
    @property
    def model(self) -> str:
        return "fake-model"

    async def complete(self, prompt: str, **kwargs: object) -> str:
        return prompt

    async def complete_with_system(self, system: str, user: str, **kwargs: object) -> str:
        return user


class ToolAgent(BaseAgent):
    async def execute(self) -> AgentResult:
        return AgentResult(output="ok", success=True)


def make_tool_definition(
    name: str,
    *,
    permission: ToolPermission = ToolPermission.READ_ONLY,
    description: str = "Reads deterministic test data.",
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=description,
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        permission=permission,
    )


async def noop_callable(inputs: dict[str, Any]) -> dict[str, Any]:
    return inputs


def test_register_tool_success() -> None:
    registry = ToolRegistry()

    registry.register(WEB_SEARCH_DEFINITION, web_search_callable)

    assert registry.resolve("web_search") == WEB_SEARCH_DEFINITION


def test_register_duplicate_tool_raises() -> None:
    registry = ToolRegistry()
    registry.register(WEB_SEARCH_DEFINITION, web_search_callable)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(WEB_SEARCH_DEFINITION, web_search_callable)


def test_resolve_unknown_tool_raises() -> None:
    registry = ToolRegistry()

    with pytest.raises(ToolNotFoundError) as exc_info:
        registry.resolve("does_not_exist")

    assert exc_info.value.tool_name == "does_not_exist"


def test_list_all_returns_registered_tools() -> None:
    registry = ToolRegistry()
    registry.register(make_tool_definition("tool_one"), noop_callable)
    registry.register(make_tool_definition("tool_two"), noop_callable)

    assert len(registry.list_all()) == 2


def test_list_by_permission_filters_correctly() -> None:
    registry = ToolRegistry()
    network_tool = make_tool_definition("network_tool", permission=ToolPermission.NETWORK)
    read_only_tool = make_tool_definition("read_only_tool", permission=ToolPermission.READ_ONLY)
    registry.register(network_tool, noop_callable)
    registry.register(read_only_tool, noop_callable)

    assert registry.list_by_permission(ToolPermission.NETWORK) == [network_tool]


@pytest.mark.asyncio
async def test_execute_with_correct_permission_succeeds() -> None:
    registry = ToolRegistry()
    registry.register(WEB_SEARCH_DEFINITION, web_search_callable)

    result = await registry.execute(
        "web_search",
        {"query": "test"},
        {ToolPermission.NETWORK},
    )

    assert isinstance(result, ToolResult)
    assert result.success is True


@pytest.mark.asyncio
async def test_execute_without_permission_raises() -> None:
    registry = ToolRegistry()
    registry.register(WEB_SEARCH_DEFINITION, web_search_callable)

    with pytest.raises(ToolPermissionError) as exc_info:
        await registry.execute(
            "web_search",
            {"query": "test"},
            {ToolPermission.READ_ONLY},
        )

    assert exc_info.value.tool_name == "web_search"
    assert exc_info.value.required == ToolPermission.NETWORK


@pytest.mark.asyncio
async def test_execute_with_empty_permissions_raises() -> None:
    registry = ToolRegistry()
    registry.register(WEB_SEARCH_DEFINITION, web_search_callable)

    with pytest.raises(ToolPermissionError):
        await registry.execute("web_search", {"query": "test"}, set())


@pytest.mark.asyncio
async def test_tool_callable_never_invoked_on_permission_failure() -> None:
    calls = 0

    async def counting_callable(inputs: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return inputs

    registry = ToolRegistry()
    registry.register(
        make_tool_definition("network_tool", permission=ToolPermission.NETWORK),
        counting_callable,
    )

    with pytest.raises(ToolPermissionError):
        await registry.execute("network_tool", {}, {ToolPermission.READ_ONLY})

    assert calls == 0


@pytest.mark.asyncio
async def test_execute_unknown_tool_raises_not_found() -> None:
    registry = ToolRegistry()

    with pytest.raises(ToolNotFoundError):
        await registry.execute("ghost_tool", {}, {ToolPermission.NETWORK})


@pytest.mark.asyncio
async def test_tool_result_success_shape() -> None:
    registry = ToolRegistry()
    registry.register(WEB_SEARCH_DEFINITION, web_search_callable)

    result = await registry.execute(
        "web_search",
        {"query": "test"},
        {ToolPermission.NETWORK},
    )

    assert result.success is True
    assert result.error is None
    assert result.tool_name == "web_search"
    assert result.execution_time_ms >= 0


@pytest.mark.asyncio
async def test_tool_result_on_callable_exception() -> None:
    async def failing_callable(inputs: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("boom")

    registry = ToolRegistry()
    registry.register(
        make_tool_definition("failing_tool", permission=ToolPermission.READ_ONLY),
        failing_callable,
    )

    result = await registry.execute("failing_tool", {}, {ToolPermission.READ_ONLY})

    assert result.success is False
    assert result.error is not None
    assert "boom" in result.error
    assert result.execution_time_ms >= 0


def test_tool_definition_is_frozen() -> None:
    definition = make_tool_definition("frozen_tool")

    with pytest.raises((ValidationError, TypeError)):
        definition.name = "mutated"


def test_base_agent_accepts_tool_registry() -> None:
    registry = ToolRegistry()
    agent = ToolAgent(
        AgentContext(agent_id="agent-1", task="task"),
        FakeProvider(),
        tool_registry=registry,
    )

    assert agent._tool_registry is registry


def test_tool_descriptions_for_prompt_format() -> None:
    registry = ToolRegistry()
    first = make_tool_definition("first_tool", description="Runs the first test tool.")
    second = make_tool_definition("second_tool", description="Runs the second test tool.")
    registry.register(first, noop_callable)
    registry.register(second, noop_callable)

    descriptions = registry.tool_descriptions_for_prompt()

    assert "first_tool" in descriptions
    assert "Runs the first test tool." in descriptions
    assert "second_tool" in descriptions
    assert "Runs the second test tool." in descriptions
