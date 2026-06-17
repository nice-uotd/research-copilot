# -*- coding: utf-8 -*-
"""Semantic Scholar 学术论文搜索工具：基于 Semantic Scholar Academic Graph API。

设计：
  - 使用 Semantic Scholar 免费 API（无需 Key）
  - 提供引用量信号，可辅助筛选高影响力论文
  - Rate limit: 100 requests / 5 min（Agent 调用频率远低于此）
  - 返回结构化 JSON 供 Agent 直接使用
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import httpx
from loguru import logger

from app.core.tools.base import BaseTool, ToolParameter

S2_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "title,authors,year,abstract,citationCount,url,venue,externalIds"


@dataclass
class ScholarPaper:
    """单篇 Semantic Scholar 论文的结构化信息。"""

    title: str
    authors: list[str]
    year: int | None
    abstract: str
    citation_count: int
    url: str
    venue: str
    paper_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ScholarSearchTool(BaseTool):
    """Semantic Scholar 学术论文搜索工具。

    适用场景：需要查找高引用量的重要论文、获取引用数据来筛选高质量参考文献、
    或补充 arXiv 搜索无法覆盖的已发表期刊/会议论文。
    """

    def __init__(self, timeout: float = 30.0) -> None:
        super().__init__()
        self.name = "scholar_search"
        self.description = (
            "搜索 Semantic Scholar 学术数据库，返回论文的标题、作者、年份、"
            "摘要、引用量和发表会议/期刊。适合查找高影响力论文、获取引用数据。"
            "与 arxiv_search 互补：Semantic Scholar 覆盖已发表论文且提供引用量排序。"
        )
        self.parameters = [
            ToolParameter(
                name="query",
                type="string",
                description="搜索查询（英文关键词或短语）",
                required=True,
            ),
            ToolParameter(
                name="max_results",
                type="integer",
                description="返回论文数量上限（1-20），默认 10",
                required=False,
            ),
            ToolParameter(
                name="year_min",
                type="integer",
                description="最早发表年份（如 2020），用于过滤旧论文",
                required=False,
            ),
        ]
        self._timeout = timeout

    async def _fetch_scholar(
        self, query: str, max_results: int, year_min: int | None
    ) -> list[ScholarPaper]:
        """调用 Semantic Scholar API 并解析 JSON 响应（含简单 retry）。"""
        params: dict[str, Any] = {
            "query": query,
            "limit": max_results,
            "fields": S2_FIELDS,
        }
        if year_min:
            params["year"] = f"{year_min}-"

        headers = {"Accept": "application/json"}

        # 简单重试：遇到 429 等待后重试 1 次
        max_attempts = 2
        last_exc: Exception | None = None

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for attempt in range(max_attempts):
                try:
                    resp = await client.get(S2_API_URL, params=params, headers=headers)
                    resp.raise_for_status()
                    break
                except httpx.HTTPStatusError as exc:
                    last_exc = exc
                    if exc.response.status_code == 429 and attempt < max_attempts - 1:
                        import asyncio
                        logger.info("Semantic Scholar 429，等待 3s 后重试...")
                        await asyncio.sleep(3)
                        continue
                    raise
            else:
                raise last_exc  # type: ignore[misc]

        data = resp.json()
        papers: list[ScholarPaper] = []

        for item in data.get("data", []):
            title = item.get("title", "")
            authors_raw = item.get("authors") or []
            authors = [a.get("name", "") for a in authors_raw[:5]]
            year = item.get("year")
            abstract = (item.get("abstract") or "")[:400]
            citation_count = item.get("citationCount", 0) or 0
            url = item.get("url", "")
            venue = item.get("venue", "") or ""
            paper_id = item.get("paperId", "")

            papers.append(
                ScholarPaper(
                    title=title,
                    authors=authors,
                    year=year,
                    abstract=abstract,
                    citation_count=citation_count,
                    url=url,
                    venue=venue,
                    paper_id=paper_id,
                )
            )

        # 按引用量降序排列
        papers.sort(key=lambda p: p.citation_count, reverse=True)
        return papers

    async def search(
        self, query: str, max_results: int = 10, year_min: int | None = None
    ) -> list[ScholarPaper]:
        """执行搜索，返回结构化论文列表（按引用量排序）。"""
        if not query.strip():
            raise ValueError("query 不能为空")
        max_results = max(1, min(max_results, 20))

        try:
            return await self._fetch_scholar(query, max_results, year_min)
        except httpx.HTTPStatusError as exc:
            logger.warning("Semantic Scholar API HTTP 错误: {}", exc)
            if exc.response.status_code == 429:
                raise RuntimeError(
                    "Semantic Scholar API 限流（100 req/5min），请稍后重试"
                ) from exc
            raise RuntimeError(
                f"Semantic Scholar API 请求失败: HTTP {exc.response.status_code}"
            ) from exc
        except Exception as exc:
            logger.exception("Semantic Scholar 搜索异常: {}", exc)
            raise RuntimeError(f"Semantic Scholar 搜索失败: {exc!s}") from exc

    async def execute(self, **kwargs: Any) -> Any:
        """Agent 工具接口：返回 JSON 字符串。"""
        query = str(kwargs.get("query", "")).strip()
        if not query:
            return json.dumps({"error": "参数 query 不能为空"}, ensure_ascii=False)

        raw_n = kwargs.get("max_results", 10)
        try:
            max_results = max(1, min(int(raw_n), 20))
        except (TypeError, ValueError):
            max_results = 10

        raw_year = kwargs.get("year_min")
        year_min = None
        if raw_year is not None:
            try:
                year_min = int(raw_year)
            except (TypeError, ValueError):
                pass

        papers = await self.search(query, max_results, year_min)
        return json.dumps(
            [p.to_dict() for p in papers],
            ensure_ascii=False,
        )
