from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class TraceEventType(StrEnum):
    AGENT_START = "agent_start"
    AGENT_COMPLETE = "agent_complete"
    AGENT_ERROR = "agent_error"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    LLM_CALL = "llm_call"
    LLM_RESULT = "llm_result"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"


class TraceEvent(BaseModel):
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    event_type: TraceEventType
    agent_id: str
    timestamp: datetime
    payload: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float | None = None
    token_count: int | None = None
    cost_usd: float | None = None

    @field_validator("timestamp")
    @classmethod
    def must_be_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("TraceEvent.timestamp must be timezone-aware UTC")
        return v


class TraceSpan(BaseModel):
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    agent_id: str
    started_at: datetime
    ended_at: datetime | None = None
    events: list[TraceEvent] = Field(default_factory=list)
    status: Literal["running", "complete", "error"] = "running"

    @field_validator("started_at")
    @classmethod
    def started_must_be_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("TraceSpan.started_at must be timezone-aware UTC")
        return v


def generate_trace_id() -> str:
    return str(uuid.uuid4())


def generate_span_id() -> str:
    return str(uuid.uuid4())
