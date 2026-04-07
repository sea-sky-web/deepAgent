from __future__ import annotations

from typing import Any

from tavily import TavilyClient

from tools.base import BaseTool


class TavilySearchTool(BaseTool):
    name = "tavily_search"
    description = "使用 Tavily 进行联网检索，返回搜索摘要"
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "要搜索的查询词"},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, api_key: str) -> None:
        self.client = TavilyClient(api_key=api_key)

    def invoke(self, tool_input: dict[str, Any]) -> str:
        query = tool_input["query"]

        response = self.client.search(
            query=query,
            search_depth="basic",
            max_results=5,
        )

        results = response.get("results", [])
        if not results:
            return "没有检索到相关结果。"

        lines: list[str] = []
        for idx, item in enumerate(results, start=1):
            title = item.get("title", "")
            content = item.get("content", "")
            url = item.get("url", "")
            lines.append(f"{idx}. 标题: {title}\n摘要: {content}\n链接: {url}")

        return "\n\n".join(lines)