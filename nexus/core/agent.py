from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import StrEnum
from time import perf_counter
from typing import TYPE_CHECKING, Any, final

from pydantic import BaseModel, ConfigDict, Field

from nexus.providers.base import LLMProvider

if TYPE_CHECKING:
    from nexus.memory.base import MemoryStore


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
    ) -> None:
        """Create an agent with runtime context, provider, and optional memory."""
        self.context = context
        self.provider = provider
        self.memory = memory
        self.state = AgentState.IDLE

    @final
    async def run(self) -> AgentResult:
        """Run the final lifecycle wrapper and always return an AgentResult."""
        self._transition(AgentState.RUNNING)
        start = perf_counter()

        try:
            result = await self.execute()
            self._transition(AgentState.DONE)
            result.duration_ms = self._duration_ms_since(start)
            return result
        except Exception as exc:
            self._transition(AgentState.FAILED)
            self.logger().exception("agent execution failed", error=str(exc))
            return AgentResult(
                output="",
                success=False,
                error=str(exc),
                duration_ms=self._duration_ms_since(start),
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
