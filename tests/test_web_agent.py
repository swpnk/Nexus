from __future__ import annotations

from typing import Any

import pytest

from nexus.agents import WebAgent
from nexus.core.agent import AgentContext, AgentResult
from nexus.providers.base import LLMProvider
from nexus.tools import ToolPermission, ToolRegistry
from nexus.tools.builtins.web_search import WEB_SEARCH_DEFINITION, web_search_callable


class FakeProvider:
    def __init__(self, response: str = "Synthesized answer from retrieved sources.") -> None:
        self.response = response
        self.system_prompt: str | None = None
        self.user_prompt: str | None = None

    @property
    def model(self) -> str:
        return "fake-model"

    async def complete(self, prompt: str, **kwargs: object) -> str:
        return self.response

    async def complete_with_system(self, system: str, user: str, **kwargs: object) -> str:
        self.system_prompt = system
        self.user_prompt = user
        return self.response


class FailingProvider:
    @property
    def model(self) -> str:
        return "failing-model"

    async def complete(self, prompt: str, **kwargs: object) -> str:
        raise RuntimeError("provider failed")

    async def complete_with_system(self, system: str, user: str, **kwargs: object) -> str:
        raise RuntimeError("provider failed")


class CleanupWebAgent(WebAgent):
    def __init__(
        self,
        context: AgentContext,
        provider: LLMProvider,
        *,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        super().__init__(context, provider, tool_registry=tool_registry)
        self.cleanup_called = False

    def cleanup(self) -> None:
        self.cleanup_called = True


def make_context(task: str = "What is Nexus?") -> AgentContext:
    return AgentContext(agent_id="web-agent-1", task=task)


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(WEB_SEARCH_DEFINITION, web_search_callable)
    return registry


@pytest.mark.asyncio
async def test_web_agent_run_returns_success_result() -> None:
    agent = WebAgent(make_context(), FakeProvider(), tool_registry=make_registry())

    result = await agent.run()

    assert isinstance(result, AgentResult)
    assert result.success is True


@pytest.mark.asyncio
async def test_web_agent_populates_evidence_after_search() -> None:
    agent = WebAgent(make_context("nexus framework"), FakeProvider(), tool_registry=make_registry())

    result = await agent.run()

    assert result.evidence
    assert "https://example.com/result-1" in result.evidence


@pytest.mark.asyncio
async def test_web_agent_reasoning_steps_reflect_retrieved_content() -> None:
    agent = WebAgent(make_context("nexus framework"), FakeProvider(), tool_registry=make_registry())

    result = await agent.run()

    assert result.reasoning_steps
    assert any("Stub result 1 for: nexus framework" in step for step in result.reasoning_steps)


@pytest.mark.asyncio
async def test_web_agent_executes_web_search_with_network_permission() -> None:
    class TrackingRegistry(ToolRegistry):
        def __init__(self) -> None:
            super().__init__()
            self.tool_name: str | None = None
            self.inputs: dict[str, Any] | None = None
            self.permissions: set[ToolPermission] | None = None

        async def execute(
            self,
            tool_name: str,
            inputs: dict[str, Any],
            agent_permissions: set[ToolPermission],
        ):
            self.tool_name = tool_name
            self.inputs = inputs
            self.permissions = agent_permissions
            return await super().execute(tool_name, inputs, agent_permissions)

    registry = TrackingRegistry()
    registry.register(WEB_SEARCH_DEFINITION, web_search_callable)
    agent = WebAgent(make_context("nexus framework"), FakeProvider(), tool_registry=registry)

    await agent.run()

    assert registry.tool_name == "web_search"
    assert registry.inputs == {"query": "nexus framework", "max_results": 5}
    assert registry.permissions == {ToolPermission.NETWORK}


@pytest.mark.asyncio
async def test_web_agent_without_tool_registry_returns_failed_result() -> None:
    agent = WebAgent(make_context(), FakeProvider())

    result = await agent.run()

    assert result.success is False
    assert result.error is not None
    assert "tool registry is required" in result.error


@pytest.mark.asyncio
async def test_web_agent_permission_violation_returns_failed_result_before_search() -> None:
    calls = 0

    async def counting_search(inputs: dict[str, Any]) -> list[dict[str, str]]:
        nonlocal calls
        calls += 1
        return await web_search_callable(inputs)

    registry = ToolRegistry()
    registry.register(WEB_SEARCH_DEFINITION, counting_search)
    agent = WebAgent(make_context(), FakeProvider(), tool_registry=registry)
    agent.agent_permissions = {ToolPermission.READ_ONLY}

    result = await agent.run()

    assert result.success is False
    assert result.error is not None
    assert "requires permission 'network'" in result.error
    assert calls == 0


@pytest.mark.asyncio
async def test_web_agent_cleanup_runs_on_provider_failure() -> None:
    agent = CleanupWebAgent(make_context(), FailingProvider(), tool_registry=make_registry())

    result = await agent.run()

    assert result.success is False
    assert result.error == "provider failed"
    assert agent.cleanup_called is True


@pytest.mark.asyncio
async def test_web_agent_empty_search_results_return_failed_result() -> None:
    async def empty_search(inputs: dict[str, Any]) -> list[dict[str, str]]:
        return []

    registry = ToolRegistry()
    registry.register(WEB_SEARCH_DEFINITION, empty_search)
    agent = WebAgent(make_context(), FakeProvider(), tool_registry=registry)

    result = await agent.run()

    assert result.success is False
    assert result.error == "web_search returned no results"
    assert result.evidence == []


@pytest.mark.asyncio
async def test_web_agent_prompt_includes_retrieved_titles_urls_and_snippets() -> None:
    provider = FakeProvider()
    agent = WebAgent(make_context("nexus framework"), provider, tool_registry=make_registry())

    await agent.run()

    assert provider.user_prompt is not None
    assert "Stub result 1 for: nexus framework" in provider.user_prompt
    assert "https://example.com/result-1" in provider.user_prompt
    assert "This is a stub snippet for result 1." in provider.user_prompt


@pytest.mark.asyncio
async def test_web_agent_output_uses_provider_response() -> None:
    provider = FakeProvider(response="Provider-authored synthesis.")
    agent = WebAgent(make_context(), provider, tool_registry=make_registry())

    result = await agent.run()

    assert result.output == "Provider-authored synthesis."
