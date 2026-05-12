from __future__ import annotations

from typing import Protocol


class LLMProvider(Protocol):
    """Structural contract implemented by all LLM provider adapters."""

    async def complete(self, prompt: str, **kwargs: object) -> str:
        """Return a text completion for a single user prompt."""
        ...

    async def complete_with_system(self, system: str, user: str, **kwargs: object) -> str:
        """Return a text completion using explicit system and user messages."""
        ...

    @property
    def model(self) -> str:
        """Return the model identifier used by this provider."""
        ...
