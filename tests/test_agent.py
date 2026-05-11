from __future__ import annotations

from datetime import UTC

import pytest

from nexus.core.agent import (
    AgentContext,
    AgentResult,
    AgentState,
    BaseAgent,
    InvalidStateTransitionError,
)
from nexus.providers.base import LLMProvider


class FakeProvider:
    @property
    def model(self) -> str:
        return "fake-model"

    async def complete(self, prompt: str, **kwargs: object) -> str:
        return f"completed: {prompt}"

    async def complete_with_system(self, system: str, user: str, **kwargs: object) -> str:
        return f"{system}: {user}"


class SuccessfulAgent(BaseAgent):
    def __init__(self, context: AgentContext, provider: LLMProvider) -> None:
        super().__init__(context, provider)
        self.cleanup_called = False

    async def execute(self) -> AgentResult:
        return AgentResult(output="ok", success=True)

    def cleanup(self) -> None:
        self.cleanup_called = True


class FailingAgent(BaseAgent):
    def __init__(self, context: AgentContext, provider: LLMProvider) -> None:
        super().__init__(context, provider)
        self.cleanup_called = False

    async def execute(self) -> AgentResult:
        raise RuntimeError("boom")

    def cleanup(self) -> None:
        self.cleanup_called = True


@pytest.fixture
def context() -> AgentContext:
    return AgentContext(agent_id="agent-1", task="test task", config={"retries": 0})


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider()


def test_valid_transition_idle_to_running_succeeds(
    context: AgentContext, provider: FakeProvider
) -> None:
    agent = SuccessfulAgent(context, provider)

    agent._transition(AgentState.RUNNING)

    assert agent.state is AgentState.RUNNING


def test_valid_transition_running_to_done_succeeds(
    context: AgentContext, provider: FakeProvider
) -> None:
    agent = SuccessfulAgent(context, provider)
    agent._transition(AgentState.RUNNING)

    agent._transition(AgentState.DONE)

    assert agent.state is AgentState.DONE


def test_valid_transition_running_to_failed_succeeds(
    context: AgentContext, provider: FakeProvider
) -> None:
    agent = SuccessfulAgent(context, provider)
    agent._transition(AgentState.RUNNING)

    agent._transition(AgentState.FAILED)

    assert agent.state is AgentState.FAILED


def test_valid_transition_running_to_cancelled_succeeds(
    context: AgentContext, provider: FakeProvider
) -> None:
    agent = SuccessfulAgent(context, provider)
    agent._transition(AgentState.RUNNING)

    agent._transition(AgentState.CANCELLED)

    assert agent.state is AgentState.CANCELLED


def test_invalid_transition_done_to_running_raises(
    context: AgentContext, provider: FakeProvider
) -> None:
    agent = SuccessfulAgent(context, provider)
    agent._transition(AgentState.RUNNING)
    agent._transition(AgentState.DONE)

    with pytest.raises(InvalidStateTransitionError):
        agent._transition(AgentState.RUNNING)


def test_invalid_transition_failed_to_running_raises(
    context: AgentContext, provider: FakeProvider
) -> None:
    agent = SuccessfulAgent(context, provider)
    agent._transition(AgentState.RUNNING)
    agent._transition(AgentState.FAILED)

    with pytest.raises(InvalidStateTransitionError):
        agent._transition(AgentState.RUNNING)


def test_invalid_transition_idle_to_done_raises(
    context: AgentContext, provider: FakeProvider
) -> None:
    agent = SuccessfulAgent(context, provider)

    with pytest.raises(InvalidStateTransitionError):
        agent._transition(AgentState.DONE)


@pytest.mark.asyncio
async def test_cleanup_called_when_execute_succeeds(
    context: AgentContext, provider: FakeProvider
) -> None:
    agent = SuccessfulAgent(context, provider)

    await agent.run()

    assert agent.cleanup_called is True


@pytest.mark.asyncio
async def test_cleanup_called_when_execute_raises(
    context: AgentContext, provider: FakeProvider
) -> None:
    agent = FailingAgent(context, provider)

    await agent.run()

    assert agent.cleanup_called is True


@pytest.mark.asyncio
async def test_run_returns_failed_agent_result_when_execute_raises(
    context: AgentContext, provider: FakeProvider
) -> None:
    agent = FailingAgent(context, provider)

    result = await agent.run()

    assert result.output == ""
    assert result.success is False
    assert result.error == "boom"


@pytest.mark.asyncio
async def test_agent_state_done_after_successful_run(
    context: AgentContext, provider: FakeProvider
) -> None:
    agent = SuccessfulAgent(context, provider)

    await agent.run()

    assert agent.state is AgentState.DONE


@pytest.mark.asyncio
async def test_agent_state_failed_after_failed_run(
    context: AgentContext, provider: FakeProvider
) -> None:
    agent = FailingAgent(context, provider)

    await agent.run()

    assert agent.state is AgentState.FAILED


def test_agent_result_created_at_is_timezone_aware_utc() -> None:
    result = AgentResult(output="ok", success=True)

    assert result.created_at.tzinfo is UTC


@pytest.mark.asyncio
async def test_duration_ms_is_positive_after_run(
    context: AgentContext, provider: FakeProvider
) -> None:
    agent = SuccessfulAgent(context, provider)

    result = await agent.run()

    assert result.duration_ms > 0


@pytest.mark.asyncio
async def test_success_true_when_execute_completes_normally(
    context: AgentContext, provider: FakeProvider
) -> None:
    agent = SuccessfulAgent(context, provider)

    result = await agent.run()

    assert result.success is True
