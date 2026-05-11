from __future__ import annotations

from typing import Any

from openai import OpenAI


class OpenAIProvider:
    def __init__(self, api_key: str, model: str, client: Any | None = None) -> None:
        self._model = model
        self._client: Any
        self._client = client or OpenAI(api_key=api_key)

    @property
    def model(self) -> str:
        return self._model

    async def complete(self, prompt: str, **kwargs: object) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return self._extract_text(response)

    async def complete_with_system(self, system: str, user: str, **kwargs: object) -> str:
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
        content = response.choices[0].message.content
        return "" if content is None else str(content)
