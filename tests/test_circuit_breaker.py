from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from nexus.agents import WebAgent
from nexus.core.circuit_breaker import (
    BreakerState,
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitOpenError,
)
from tests.test_web_agent import FakeProvider, make_context, make_registry


async def succeed() -> str:
    return "ok"


async def fail() -> str:
    raise RuntimeError("failure")


@pytest.mark.asyncio
async def test_breaker_starts_closed() -> None:
    breaker = CircuitBreaker(name="test", rolling_window=5, failure_threshold=0.5)
    assert breaker.state == BreakerState.CLOSED


@pytest.mark.asyncio
async def test_breaker_stays_closed_below_threshold() -> None:
    breaker = CircuitBreaker(name="test", rolling_window=4, failure_threshold=0.5)

    await breaker.call(succeed)
    await breaker.call(succeed)
    await breaker.call(succeed)
    with pytest.raises(RuntimeError):
        await breaker.call(fail)

    assert breaker.state == BreakerState.CLOSED


@pytest.mark.asyncio
async def test_breaker_trips_to_open() -> None:
    breaker = CircuitBreaker(name="test", rolling_window=4, failure_threshold=0.5)

    with pytest.raises(RuntimeError):
        await breaker.call(fail)
    with pytest.raises(RuntimeError):
        await breaker.call(fail)
    await breaker.call(succeed)
    await breaker.call(succeed)

    assert breaker.state == BreakerState.OPEN


@pytest.mark.asyncio
async def test_open_breaker_raises_immediately() -> None:
    breaker = CircuitBreaker(name="test", rolling_window=2, failure_threshold=0.5)
    breaker._transition(BreakerState.OPEN)
    breaker._opened_at = datetime.now(UTC)

    with pytest.raises(CircuitOpenError) as exc_info:
        await breaker.call(succeed)

    assert exc_info.value.breaker_name == "test"


@pytest.mark.asyncio
async def test_transitions_to_half_open_after_timeout() -> None:
    breaker = CircuitBreaker(
        name="test",
        rolling_window=2,
        failure_threshold=0.5,
        recovery_timeout=0.01,
    )
    with pytest.raises(RuntimeError):
        await breaker.call(fail)
    with pytest.raises(RuntimeError):
        await breaker.call(fail)
    assert breaker.state == BreakerState.OPEN

    await asyncio.sleep(0.02)
    result = await breaker.call(succeed)

    assert result == "ok"
    assert breaker.state == BreakerState.CLOSED


@pytest.mark.asyncio
async def test_successful_probe_closes_breaker() -> None:
    breaker = CircuitBreaker(
        name="test",
        rolling_window=2,
        failure_threshold=0.5,
        recovery_timeout=0.01,
    )
    with pytest.raises(RuntimeError):
        await breaker.call(fail)
    with pytest.raises(RuntimeError):
        await breaker.call(fail)

    await asyncio.sleep(0.02)
    await breaker.call(succeed)

    assert breaker.state == BreakerState.CLOSED


@pytest.mark.asyncio
async def test_failed_probe_reopens_breaker() -> None:
    breaker = CircuitBreaker(
        name="test",
        rolling_window=2,
        failure_threshold=0.5,
        recovery_timeout=0.01,
    )
    with pytest.raises(RuntimeError):
        await breaker.call(fail)
    with pytest.raises(RuntimeError):
        await breaker.call(fail)

    await asyncio.sleep(0.02)
    with pytest.raises(RuntimeError):
        await breaker.call(fail)

    assert breaker.state == BreakerState.OPEN


def test_invalid_transition_raises() -> None:
    breaker = CircuitBreaker(name="test")
    with pytest.raises(ValueError):
        breaker._transition(BreakerState.HALF_OPEN)


@pytest.mark.asyncio
async def test_web_agent_handles_open_circuit() -> None:
    breaker = CircuitBreaker(name="web_search", rolling_window=2, failure_threshold=0.5)
    breaker._transition(BreakerState.OPEN)
    breaker._opened_at = datetime.now(UTC)

    agent = WebAgent(
        make_context(),
        FakeProvider(),
        tool_registry=make_registry(),
        circuit_breaker=breaker,
    )

    result = await agent.run()

    assert result.success is False
    assert result.error is not None
    assert "Circuit breaker open" in result.error


def test_registry_snapshot() -> None:
    registry = CircuitBreakerRegistry()
    breaker = CircuitBreaker(name="search_api")
    registry.register(breaker)
    snap = registry.snapshot()
    assert snap["search_api"]["state"] == "closed"
    assert snap["search_api"]["failure_rate"] == 0.0
