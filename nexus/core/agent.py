from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import StrEnum
from time import perf_counter
from typing import TYPE_CHECKING, Any, final

from pydantic import BaseModel, ConfigDict, Field

from nexus.observability.schema import TraceEvent, TraceEventType, generate_trace_id
from nexus.observability.tracer import NoOpTracer, Tracer
from nexus.providers.base import LLMProvider

if TYPE_CHECKING:
    from nexus.memory.base import MemoryStore
    from nexus.tools.registry import ToolRegistry


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(UTC)


class AgentState(StrEnum):
    """Lifecycle states shared by all Nexus agents."""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


VALID_TRANSITIONS: dict[AgentState, tuple[AgentState, ...]] = {
    AgentState.IDLE: (AgentState.RUNNING, AgentState.CANCELLED),
    AgentState.RUNNING: (AgentState.DONE, AgentState.FAILED, AgentState.CANCELLED),
    AgentState.DONE: (),
    AgentState.FAILED: (),
    AgentState.CANCELLED: (),
}


class InvalidStateTransitionError(RuntimeError):
    """Raised when an agent attempts a transition outside VALID_TRANSITIONS."""

    def __init__(self, current: AgentState, new: AgentState) -> None:
        super().__init__(f"Invalid agent state transition: {current.value} -> {new.value}")


class AgentContext(BaseModel):
    """Immutable run context supplied to an agent at construction time."""

    model_config = ConfigDict(extra="forbid")

    agent_id: str
    task: str
    config: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    trace_id: str | None = None
    root_span_id: str | None = None


class AgentResult(BaseModel):
    """Normalized outcome returned by every agent run."""

    model_config = ConfigDict(extra="forbid")

    output: str
    success: bool
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0
    created_at: datetime = Field(default_factory=utc_now)
    reasoning_steps: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class BaseAgent(ABC):
    """Base class that owns the agent lifecycle around subclass execution."""

    def __init__(
        self,
        context: AgentContext,
        provider: LLMProvider,
        memory: MemoryStore | None = None,
        tool_registry: ToolRegistry | None = None,
        *,
        tracer: Tracer | None = None,
    ) -> None:
        """Create an agent with runtime context and injected dependencies."""
        self.context = context
        self.provider = provider
        self.memory = memory
        self._tool_registry = tool_registry
        self.tracer: Tracer = tracer if tracer is not None else NoOpTracer()
        self.state = AgentState.IDLE

    @final
    async def run(self) -> AgentResult:
        """Run the final lifecycle wrapper and always return an AgentResult."""
        self._transition(AgentState.RUNNING)
        trace_id = generate_trace_id()
        root_span_id = self.tracer.start_span(
            trace_id=trace_id,
            agent_id=self.context.agent_id,
            parent_span_id=None,
        )
        self.context = self.context.model_copy(
            update={"trace_id": trace_id, "root_span_id": root_span_id}
        )
        self.tracer.record_event(
            root_span_id,
            TraceEvent(
                trace_id=trace_id,
                span_id=root_span_id,
                event_type=TraceEventType.AGENT_START,
                agent_id=self.context.agent_id,
                timestamp=utc_now(),
                payload={"input": str(self.context.task)[:500]},
            ),
        )
        start = perf_counter()

        try:
            result = await self.execute()
            duration_ms = self._duration_ms_since(start)
            self.tracer.record_event(
                root_span_id,
                TraceEvent(
                    trace_id=trace_id,
                    span_id=root_span_id,
                    event_type=TraceEventType.AGENT_COMPLETE,
                    agent_id=self.context.agent_id,
                    timestamp=utc_now(),
                    duration_ms=duration_ms,
                    payload={"success": result.success},
                ),
            )
            self.tracer.end_span(root_span_id, "complete", duration_ms)
            self._transition(AgentState.DONE)
            result.duration_ms = duration_ms
            return result
        except Exception as exc:
            duration_ms = self._duration_ms_since(start)
            self.tracer.record_event(
                root_span_id,
                TraceEvent(
                    trace_id=trace_id,
                    span_id=root_span_id,
                    event_type=TraceEventType.AGENT_ERROR,
                    agent_id=self.context.agent_id,
                    timestamp=utc_now(),
                    duration_ms=duration_ms,
                    payload={
                        "error_type": type(exc).__name__,
                        "error_msg": str(exc)[:500],
                    },
                ),
            )
            self.tracer.end_span(root_span_id, "error", duration_ms)
            self._transition(AgentState.FAILED)
            self.logger().exception("agent execution failed", error=str(exc))
            return AgentResult(
                output="",
                success=False,
                error=str(exc),
                duration_ms=duration_ms,
            )
        finally:
            self.cleanup()

    @abstractmethod
    async def execute(self) -> AgentResult:
        """Perform agent-specific work inside the framework-owned lifecycle."""
        raise NotImplementedError

    def cleanup(self) -> None:
        """Release resources after execution succeeds or fails."""
        return None

    def _transition(self, new_state: AgentState) -> None:
        """Move to a new state if VALID_TRANSITIONS allows it."""
        if new_state not in VALID_TRANSITIONS[self.state]:
            raise InvalidStateTransitionError(self.state, new_state)
        self.state = new_state

    def logger(self) -> Any:
        """Return a logger bound to this agent's context."""
        from nexus.observability.logging import get_agent_logger

        return get_agent_logger(self)

    @staticmethod
    def _duration_ms_since(start: float) -> float:
        """Return elapsed milliseconds since a perf_counter start value."""
        return (perf_counter() - start) * 1000
