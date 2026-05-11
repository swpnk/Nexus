from __future__ import annotations

from types import SimpleNamespace

import pytest

from nexus.config.settings import Settings
from nexus.providers.anthropic import AnthropicProvider
from nexus.providers.factory import get_provider
from nexus.providers.openai import OpenAIProvider


class FakeAnthropicMessages:
    def create(self, **kwargs: object) -> object:
        return SimpleNamespace(content=[SimpleNamespace(text="anthropic response")])


class FakeAnthropicClient:
    def __init__(self) -> None:
        self.messages = FakeAnthropicMessages()


class FakeOpenAICompletions:
    def create(self, **kwargs: object) -> object:
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="openai response"))]
        )


class FakeOpenAIChat:
    def __init__(self) -> None:
        self.completions = FakeOpenAICompletions()


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.chat = FakeOpenAIChat()


@pytest.fixture
def settings() -> Settings:
    return Settings(
        ANTHROPIC_API_KEY="anthropic-key",
        OPENAI_API_KEY="openai-key",
        DEFAULT_PROVIDER="anthropic",
        DEFAULT_MODEL="test-model",
    )


@pytest.mark.asyncio
async def test_anthropic_provider_complete_returns_string() -> None:
    provider = AnthropicProvider(api_key="key", model="test-model", client=FakeAnthropicClient())

    result = await provider.complete("hello")

    assert result == "anthropic response"


@pytest.mark.asyncio
async def test_openai_provider_complete_returns_string() -> None:
    provider = OpenAIProvider(api_key="key", model="test-model", client=FakeOpenAIClient())

    result = await provider.complete("hello")

    assert result == "openai response"


def test_factory_returns_anthropic_provider_for_anthropic(settings: Settings) -> None:
    provider = get_provider("anthropic", settings)

    assert isinstance(provider, AnthropicProvider)


def test_factory_returns_openai_provider_for_openai(settings: Settings) -> None:
    provider = get_provider("openai", settings)

    assert isinstance(provider, OpenAIProvider)


def test_factory_raises_value_error_for_unknown_provider_name(settings: Settings) -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("local", settings)
