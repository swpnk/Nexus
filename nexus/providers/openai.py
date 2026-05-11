from __future__ import annotations

from typing import Any

from openai import OpenAI


class OpenAIProvider:
    """LLMProvider adapter for OpenAI chat completions."""

    def __init__(self, api_key: str, model: str, client: Any | None = None) -> None:
        """Create an OpenAI provider with an optional injected test client."""
        self._model = model
        self._client: Any
        self._client = client or OpenAI(api_key=api_key)

    @property
    def model(self) -> str:
        """Return the configured OpenAI model name."""
        return self._model

    async def complete(self, prompt: str, **kwargs: object) -> str:
        """Return text for a single user prompt."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return self._extract_text(response)

    async def complete_with_system(self, system: str, user: str, **kwargs: object) -> str:
        """Return text for a user prompt with an OpenAI system message."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
        return self._extract_text(response)

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract the first assistant message from an OpenAI chat response."""
        content = response.choices[0].message.content
        return "" if content is None else str(content)
