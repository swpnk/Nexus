from __future__ import annotations

from nexus.config.settings import Settings
from nexus.providers.anthropic import AnthropicProvider
from nexus.providers.base import LLMProvider
from nexus.providers.openai import OpenAIProvider


def get_provider(name: str, settings: Settings) -> LLMProvider:
    normalized_name = name.lower()
    if normalized_name == "anthropic":
        return AnthropicProvider(
            api_key=settings.anthropic_api_key,
            model=settings.default_model,
        )
    if normalized_name == "openai":
        return OpenAIProvider(
            api_key=settings.openai_api_key,
            model=settings.default_model,
        )
    raise ValueError(f"Unknown provider: {name}")
