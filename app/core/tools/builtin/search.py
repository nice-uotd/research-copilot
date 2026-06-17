# -*- coding: utf-8 -*-
"""Web 搜索工具：Tavily 优先 + DDGS 回退（双提供商策略）。"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from loguru import logger

from app.core.tools.base import BaseTool, ToolParameter


@dataclass
class WebSearchResult:
    """统一的 web 搜索单条结果（提供商无关）。"""

    title: str
    url: str
    content: str
    score: float = 0.0
    source: str = ""  # tavily | ddgs

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "score": self.score,
            "source": self.source,
        }


class WebSearchTool(BaseTool):
    """Web 搜索工具：Tavily（优先，需 TAVILY_API_KEY）→ DDGS（回退，无需 key）。

    设计：
      - Tavily 专为 LLM 优化，返回更短摘要、相关性评分；
      - DDGS 走多搜索后端聚合，无需付费，但无相关性分；
      - 提供商在 search() 内部决策，调用方无感。
    """

    def __init__(
        self,
        tavily_api_key: str | None = None,
        max_results: int = 5,
        timeout: float = 15.0,
    ) -> None:
        super().__init__()
        self.name = "web_search"
        self.description = (
            "实时联网搜索，返回相关网页的标题、URL 与摘要。"
            "适合查询时效性信息或知识库未覆盖内容。"
        )
        self.parameters = [
            ToolParameter(
                name="query",
                type="string",
                description="搜索查询（自然语言或关键词）",
                required=True,
            ),
            ToolParameter(
                name="max_results",
                type="integer",
                description="返回结果数量上限（1-10）",
                required=False,
            ),
        ]
        self._tavily_key = (tavily_api_key or "").strip()
        self._max_results = max(1, min(max_results, 10))
        self._timeout = timeout
        self._tavily_client: Any = None  # 懒加载

    @property
    def has_tavily(self) -> bool:
        return bool(self._tavily_key)

    async def search(
        self,
        query: str,
        max_results: int | None = None,
    ) -> list[WebSearchResult]:
        """执行搜索，返回结构化结果。Tavily 优先，失败回退 DDGS。"""
        if not query.strip():
            raise ValueError("query 不能为空")
        n = max(1, min(max_results or self._max_results, 10))

        if self.has_tavily:
            try:
                results = await self._search_tavily(query, n)
                if results:
                    return results
                logger.warning("Tavily 返回空结果，回退到 DDGS")
            except Exception as exc:
                logger.warning("Tavily 调用失败，回退 DDGS: {}", exc)

        return await self._search_ddgs(query, n)

    async def _search_tavily(self, query: str, n: int) -> list[WebSearchResult]:
        from tavily import TavilyClient

        if self._tavily_client is None:
            self._tavily_client = TavilyClient(api_key=self._tavily_key)

        def _sync_call() -> dict[str, Any]:
            return self._tavily_client.search(
                query=query,
                max_results=n,
                search_depth="basic",
            )

        raw = await asyncio.to_thread(_sync_call)
        out: list[WebSearchResult] = []
        for item in raw.get("results", []):
            out.append(
                WebSearchResult(
                    title=str(item.get("title", "")),
                    url=str(item.get("url", "")),
                    content=str(item.get("content", "")),
                    score=float(item.get("score", 0.0)),
                    source="tavily",
                )
            )
        return out

    async def _search_ddgs(self, query: str, n: int) -> list[WebSearchResult]:
        from ddgs import DDGS

        def _sync_call() -> list[dict[str, Any]]:
            return list(DDGS().text(query, max_results=n, timeout=int(self._timeout)))

        try:
            raw = await asyncio.to_thread(_sync_call)
        except Exception as exc:
            logger.exception("DDGS 也失败了: {}", exc)
            raise RuntimeError(f"web 搜索全部不可用: {exc!s}") from exc

        out: list[WebSearchResult] = []
        for i, item in enumerate(raw):
            out.append(
                WebSearchResult(
                    title=str(item.get("title", "")),
                    url=str(item.get("href", item.get("url", ""))),
                    content=str(item.get("body", item.get("content", ""))),
                    score=1.0 / (i + 1),  # 排名倒数作为伪相关度
                    source="ddgs",
                )
            )
        return out

    async def execute(self, **kwargs: Any) -> Any:
        """ReAct 入口：返回 JSON 字符串供 LLM 解析。"""
        query = str(kwargs.get("query", "")).strip()
        raw_n = kwargs.get("max_results")
        if isinstance(raw_n, str):
            try:
                raw_n = int(raw_n)
            except ValueError:
                raw_n = None
        results = await self.search(query, raw_n if isinstance(raw_n, int) else None)
        return json.dumps(
            [r.to_dict() for r in results],
            ensure_ascii=False,
        )
