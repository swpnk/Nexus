from __future__ import annotations

from typing import Protocol


class LLMProvider(Protocol):
    async def complete(self, prompt: str, **kwargs: object) -> str: ...

    async def complete_with_system(self, system: str, user: str, **kwargs: object) -> str: ...

    @property
    def model(self) -> str: ...
