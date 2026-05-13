from __future__ import annotations

from typing import Any

from nexus.core.agent import AgentContext, AgentResult, BaseAgent
from nexus.memory.base import MemoryStore
from nexus.providers.base import LLMProvider
from nexus.tools import ToolPermission, ToolRegistry


class WebAgent(BaseAgent):
    """Agent that retrieves web evidence and synthesizes an answer from it."""

    def __init__(
        self,
        context: AgentContext,
        provider: LLMProvider,
        *,
        memory: MemoryStore | None = None,
        tool_registry: ToolRegistry | None = None,
        max_results: int = 5,
    ) -> None:
        """Create a web agent with network tool permission."""
        super().__init__(
            context,
            provider,
            memory=memory,
            tool_registry=tool_registry,
        )
        self.max_results = max_results
        self.agent_permissions = {ToolPermission.NETWORK}

    async def execute(self) -> AgentResult:
        """Retrieve web results, reason over them, and surface cited evidence."""
        if self._tool_registry is None:
            raise RuntimeError("tool registry is required for WebAgent")

        tool_result = await self._tool_registry.execute(
            "web_search",
            {"query": self.context.task, "max_results": self.max_results},
            self.agent_permissions,
        )
        if not tool_result.success:
            return AgentResult(
                output="",
                success=False,
                error=tool_result.error or "web_search failed",
            )

        results = self._normalize_results(tool_result.output)
        if not results:
            raise RuntimeError("web_search returned no results")

        system_prompt = (
            "You are a research agent. Answer only from the retrieved sources. "
            "Ground every claim in the provided titles, URLs, and snippets."
        )
        user_prompt = self._build_user_prompt(self.context.task, results)
        synthesis = await self.provider.complete_with_system(system_prompt, user_prompt)

        return AgentResult(
            output=synthesis,
            success=True,
            reasoning_steps=self._build_reasoning_steps(results),
            evidence=[result["url"] for result in results],
            metadata={"tool": "web_search", "result_count": len(results)},
        )

    @staticmethod
    def _normalize_results(output: Any) -> list[dict[str, str]]:
        """Convert tool output into title, URL, snippet dictionaries."""
        if not isinstance(output, list):
            raise RuntimeError("web_search returned an invalid result shape")

        normalized: list[dict[str, str]] = []
        for item in output:
            if not isinstance(item, dict):
                raise RuntimeError("web_search returned an invalid result shape")
            normalized.append(
                {
                    "title": str(item.get("title", "")),
                    "url": str(item.get("url", "")),
                    "snippet": str(item.get("snippet", "")),
                }
            )
        return normalized

    @staticmethod
    def _build_user_prompt(query: str, results: list[dict[str, str]]) -> str:
        """Build the grounded synthesis prompt from retrieved results."""
        sources = []
        for index, result in enumerate(results, start=1):
            sources.append(
                "\n".join(
                    [
                        f"Source {index}: {result['title']}",
                        f"URL: {result['url']}",
                        f"Snippet: {result['snippet']}",
                    ]
                )
            )
        return f"Question: {query}\n\nRetrieved sources:\n\n" + "\n\n".join(sources)

    @staticmethod
    def _build_reasoning_steps(results: list[dict[str, str]]) -> list[str]:
        """Create a deterministic reasoning audit trail from retrieved evidence."""
        return [
            f"Retrieved {result['title']} from {result['url']}: {result['snippet']}"
            for result in results
        ]
