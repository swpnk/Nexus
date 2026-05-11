from __future__ import annotations

from typing import Any

from anthropic import Anthropic


class AnthropicProvider:
    def __init__(self, api_key: str, model: str, client: Any | None = None) -> None:
        self._model = model
        self._client: Any
        self._client = client or Anthropic(api_key=api_key)

    @property
    def model(self) -> str:
        return self._model

    async def complete(self, prompt: str, **kwargs: object) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=kwargs.pop("max_tokens", 1024),
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return self._extract_text(response)

    async def complete_with_system(self, system: str, user: str, **kwargs: object) -> str:
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
        chunks = []
        for block in response.content:
            text = getattr(block, "text", None)
            if text is not None:
                chunks.append(text)
        return "".join(chunks)
