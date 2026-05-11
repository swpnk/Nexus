from __future__ import annotations

from datetime import UTC

import pytest
from pydantic import ValidationError

from nexus.config.settings import Settings
from nexus.core.agent import AgentContext, AgentResult


def test_missing_anthropic_api_key_raises_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("DEFAULT_PROVIDER", "anthropic")
    monkeypatch.setenv("DEFAULT_MODEL", "test-model")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_missing_openai_api_key_raises_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DEFAULT_PROVIDER", "anthropic")
    monkeypatch.setenv("DEFAULT_MODEL", "test-model")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_agent_context_created_at_is_timezone_aware_utc() -> None:
    context = AgentContext(agent_id="agent-1", task="task")

    assert context.created_at.tzinfo is UTC


def test_agent_result_created_at_is_timezone_aware_utc() -> None:
    result = AgentResult(output="ok", success=True)

    assert result.created_at.tzinfo is UTC
