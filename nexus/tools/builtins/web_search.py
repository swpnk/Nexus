from nexus.tools.base import ToolDefinition, ToolPermission

WEB_SEARCH_DEFINITION = ToolDefinition(
    name="web_search",
    description=(
        "Searches the web for a query and returns a list of results with titles, URLs, "
        "and snippets."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query to execute"},
            "max_results": {
                "type": "integer",
                "default": 5,
                "description": "Maximum number of results to return",
            },
        },
        "required": ["query"],
    },
    output_schema={
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "url": {"type": "string"},
                "snippet": {"type": "string"},
            },
        },
    },
    permission=ToolPermission.NETWORK,
    version="1.0.0",
)


async def web_search_callable(inputs: dict[str, object]) -> list[dict[str, str]]:
    """Return deterministic stub search results until Day 3 adds real search."""
    query = str(inputs.get("query", ""))
    raw_max_results = inputs.get("max_results", 5)
    max_results = raw_max_results if isinstance(raw_max_results, int) else int(str(raw_max_results))
    return [
        {
            "title": f"Stub result {index + 1} for: {query}",
            "url": f"https://example.com/result-{index + 1}",
            "snippet": (
                f"This is a stub snippet for result {index + 1}. "
                "Real implementation in Day 3."
            ),
        }
        for index in range(min(max_results, 3))
    ]
