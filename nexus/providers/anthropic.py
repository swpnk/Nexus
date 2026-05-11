from __future__ import annotations

from typing import Any

from anthropic import Anthropic


class AnthropicProvider:
    """LLMProvider adapter for Anthropic Messages API completions."""

    def __init__(self, api_key: str, model: str, client: Any | None = None) -> None:
        """Create an Anthropic provider with an optional injected test client."""
        self._model = model
        self._client: Any
        self._client = client or Anthropic(api_key=api_key)

    @property
    def model(self) -> str:
        """Return the configured Anthropic model name."""
        return self._model

    async def complete(self, prompt: str, **kwargs: object) -> str:
        """Return text for a single user prompt."""
        response = self._client.messages.create(
            model=self._model,
            max_tokens=kwargs.pop("max_tokens", 1024),
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return self._extract_text(response)

    async def complete_with_system(self, system: str, user: str, **kwargs: object) -> str:
        """Return text for a user prompt with an Anthropic system prompt."""
        response = self._client.messages.create(
            model=self._model,
            max_tokens=kwargs.pop("max_tokens", 1024),
            system=system,
            messages=[{"role": "user", "content": user}],
            **kwargs,
        )
        return self._extract_text(response)

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract concatenated text blocks from an Anthropic response."""
        chunks = []
        for block in response.content:
            text = getattr(block, "text", None)
            if text is not None:
                chunks.append(text)
        return "".join(chunks)
