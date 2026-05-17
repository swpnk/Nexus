from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from nexus.observability.tracer import Tracer

T = TypeVar("T")


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


VALID_BREAKER_TRANSITIONS: dict[BreakerState, set[BreakerState]] = {
    BreakerState.CLOSED: {BreakerState.OPEN},
    BreakerState.OPEN: {BreakerState.HALF_OPEN},
    BreakerState.HALF_OPEN: {BreakerState.CLOSED, BreakerState.OPEN},
}


class CircuitOpenError(Exception):
    """Raised when a call is attempted against an OPEN circuit breaker."""

    def __init__(self, breaker_name: str, retry_after: datetime) -> None:
        self.breaker_name = breaker_name
        self.retry_after = retry_after
        super().__init__(
            f"Circuit '{breaker_name}' is OPEN. Retry after {retry_after.isoformat()}"
        )


class CircuitBreaker:
    """
    Three-state fault isolation wrapper for async callables.

    States: CLOSED (normal) → OPEN (tripped) → HALF_OPEN (probing) → CLOSED

    The breaker is a generic callable wrapper. It does not know about
    tools, LLMs, or HTTP. Callers inject it; tools and providers use it.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: float = 0.5,
        rolling_window: int = 10,
        recovery_timeout: float = 60.0,
        tracer: Tracer | None = None,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.rolling_window = rolling_window
        self.recovery_timeout = recovery_timeout
        self._tracer = tracer

        self._state = BreakerState.CLOSED
        self._results: deque[bool] = deque(maxlen=rolling_window)
        self._opened_at: datetime | None = None
        self._lock = asyncio.Lock()
        self._obs_trace_id: str | None = None
        self._obs_span_id: str | None = None
        if tracer is not None:
            from nexus.observability.schema import generate_trace_id

            self._obs_trace_id = generate_trace_id()
            self._obs_span_id = tracer.start_span(
                self._obs_trace_id,
                f"circuit_breaker:{name}",
            )

    def _transition(self, new_state: BreakerState) -> None:
        """Validate and apply a state transition. Emit trace event."""
        if new_state not in VALID_BREAKER_TRANSITIONS[self._state]:
            raise ValueError(f"Invalid breaker transition: {self._state} → {new_state}")
        old_state = self._state
        self._state = new_state
        if new_state == BreakerState.OPEN:
            self._opened_at = datetime.now(UTC)
        elif new_state == BreakerState.CLOSED:
            self._opened_at = None
        self._emit_transition_event(old_state, new_state)

    def _emit_transition_event(self, old_state: BreakerState, new_state: BreakerState) -> None:
        if self._tracer is None or self._obs_span_id is None or self._obs_trace_id is None:
            return
        from nexus.observability.schema import TraceEvent, TraceEventType

        self._tracer.record_event(
            self._obs_span_id,
            TraceEvent(
                trace_id=self._obs_trace_id,
                span_id=self._obs_span_id,
                event_type=TraceEventType.TOOL_CALL,
                agent_id=f"circuit_breaker:{self.name}",
                timestamp=datetime.now(UTC),
                payload={
                    "circuit_breaker_transition": True,
                    "breaker": self.name,
                    "from_state": old_state.value,
                    "to_state": new_state.value,
                    "failure_rate": self.failure_rate,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            ),
        )

    def _record(self, success: bool) -> None:
        """Append outcome to rolling window."""
        self._results.append(success)

    def _should_trip(self) -> bool:
        """True if failure rate in current window meets or exceeds threshold."""
        if len(self._results) < self.rolling_window:
            return False
        return self.failure_rate >= self.failure_threshold

    def _recovery_elapsed(self) -> bool:
        """True if recovery_timeout has passed since the breaker opened."""
        if self._opened_at is None:
            return False
        elapsed = (datetime.now(UTC) - self._opened_at).total_seconds()
        return elapsed >= self.recovery_timeout

    @property
    def state(self) -> BreakerState:
        return self._state

    @property
    def failure_rate(self) -> float:
        if not self._results:
            return 0.0
        failures = sum(1 for result in self._results if not result)
        return failures / len(self._results)

    async def call(
        self,
        fn: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute fn if CLOSED or probing (HALF_OPEN).
        Reject immediately with CircuitOpenError if OPEN.
        Record outcome. Transition if thresholds crossed.

        Thread-safe via asyncio.Lock.
        """
        async with self._lock:
            if self._state == BreakerState.OPEN:
                if self._recovery_elapsed():
                    self._transition(BreakerState.HALF_OPEN)
                else:
                    assert self._opened_at is not None
                    retry_at = self._opened_at + timedelta(seconds=self.recovery_timeout)
                    raise CircuitOpenError(breaker_name=self.name, retry_after=retry_at)

        try:
            result = await fn(*args, **kwargs)
            success = True
        except Exception:
            success = False
            raise
        finally:
            async with self._lock:
                self._record(success)
                if self._state == BreakerState.HALF_OPEN:
                    if success:
                        self._transition(BreakerState.CLOSED)
                        self._results.clear()
                    else:
                        self._transition(BreakerState.OPEN)
                elif self._state == BreakerState.CLOSED and self._should_trip():
                    self._transition(BreakerState.OPEN)

        return result


class CircuitBreakerRegistry:
    """
    Holds named CircuitBreaker instances.
    Provides a single view of all breaker states for operators.
    """

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}

    def register(self, breaker: CircuitBreaker) -> None:
        self._breakers[breaker.name] = breaker

    def get(self, name: str) -> CircuitBreaker | None:
        return self._breakers.get(name)

    def get_or_create(self, name: str, **kwargs: Any) -> CircuitBreaker:
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name=name, **kwargs)
        return self._breakers[name]

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Return current state of all breakers. For observability endpoints."""
        return {
            name: {
                "state": breaker.state.value,
                "failure_rate": round(breaker.failure_rate, 3),
            }
            for name, breaker in self._breakers.items()
        }
