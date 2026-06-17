# -*- coding: utf-8 -*-
"""arXiv 学术论文搜索工具：基于 arXiv REST API（Atom feed）。

设计：
  - 使用 arXiv 官方 API（http://export.arxiv.org/api/query）
  - 无需 API Key，免费无限制（建议间隔 3s，本工具由 Agent 调用频率低不成问题）
  - 解析 Atom XML feed，提取论文元数据
  - 返回结构化 JSON 供 Agent 直接使用
"""

from __future__ import annotations

import asyncio
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import quote

import httpx
from loguru import logger

from app.core.tools.base import BaseTool, ToolParameter

# arXiv Atom feed 命名空间
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

ARXIV_API_URL = "https://export.arxiv.org/api/query"


@dataclass
class ArxivPaper:
    """单篇 arXiv 论文的结构化信息。"""

    title: str
    authors: list[str]
    year: int
    abstract: str
    arxiv_id: str
    url: str
    categories: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ArxivSearchTool(BaseTool):
    """arXiv 学术论文搜索工具。

    适用场景：用户需要查找某个研究主题的相关论文、了解某领域最新进展、
    或为 Related Work 收集参考文献。
    """

    def __init__(self, timeout: float = 30.0) -> None:
        super().__init__()
        self.name = "arxiv_search"
        self.description = (
            "搜索 arXiv 学术论文数据库，返回相关论文的标题、作者、年份、"
            "摘要和链接。适合查找某研究主题的相关工作、最新预印本。"
        )
        self.parameters = [
            ToolParameter(
                name="query",
                type="string",
                description="搜索查询（英文关键词或短语，如 'retrieval augmented generation'）",
                required=True,
            ),
            ToolParameter(
                name="max_results",
                type="integer",
                description="返回论文数量上限（1-20），默认 10",
                required=False,
            ),
            ToolParameter(
                name="sort_by",
                type="string",
                description="排序方式：relevance（相关性）或 lastUpdatedDate（最新优先），默认 relevance",
                required=False,
            ),
        ]
        self._timeout = timeout

    async def _fetch_arxiv(
        self, query: str, max_results: int, sort_by: str
    ) -> list[ArxivPaper]:
        """调用 arXiv API 并解析 Atom feed。"""
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": "descending",
        }

        async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
            resp = await client.get(ARXIV_API_URL, params=params)
            resp.raise_for_status()

        root = ET.fromstring(resp.text)
        papers: list[ArxivPaper] = []

        for entry in root.findall("atom:entry", _NS):
            # 标题（去换行）
            title_el = entry.find("atom:title", _NS)
            title = re.sub(r"\s+", " ", (title_el.text or "").strip()) if title_el is not None else ""

            # 作者列表
            authors = []
            for author_el in entry.findall("atom:author", _NS):
                name_el = author_el.find("atom:name", _NS)
                if name_el is not None and name_el.text:
                    authors.append(name_el.text.strip())

            # 发布年份
            published_el = entry.find("atom:published", _NS)
            year = 2024
            if published_el is not None and published_el.text:
                match = re.match(r"(\d{4})", published_el.text)
                if match:
                    year = int(match.group(1))

            # 摘要（截断到 400 字符）
            summary_el = entry.find("atom:summary", _NS)
            abstract = ""
            if summary_el is not None and summary_el.text:
                abstract = re.sub(r"\s+", " ", summary_el.text.strip())[:400]

            # arXiv ID 和 URL
            id_el = entry.find("atom:id", _NS)
            url = (id_el.text or "").strip() if id_el is not None else ""
            arxiv_id = url.split("/abs/")[-1] if "/abs/" in url else url

            # 分类
            categories = []
            for cat_el in entry.findall("atom:category", _NS):
                term = cat_el.get("term", "")
                if term:
                    categories.append(term)

            papers.append(
                ArxivPaper(
                    title=title,
                    authors=authors[:5],  # 最多保留 5 个作者
                    year=year,
                    abstract=abstract,
                    arxiv_id=arxiv_id,
                    url=url,
                    categories=categories[:3],
                )
            )

        return papers

    async def search(
        self, query: str, max_results: int = 10, sort_by: str = "relevance"
    ) -> list[ArxivPaper]:
        """执行搜索，返回结构化论文列表。"""
        if not query.strip():
            raise ValueError("query 不能为空")
        max_results = max(1, min(max_results, 20))
        if sort_by not in ("relevance", "lastUpdatedDate"):
            sort_by = "relevance"

        try:
            return await self._fetch_arxiv(query, max_results, sort_by)
        except httpx.HTTPStatusError as exc:
            logger.warning("arXiv API HTTP 错误: {}", exc)
            raise RuntimeError(f"arXiv API 请求失败: HTTP {exc.response.status_code}") from exc
        except ET.ParseError as exc:
            logger.warning("arXiv XML 解析失败: {}", exc)
            raise RuntimeError("arXiv 返回数据解析失败") from exc
        except Exception as exc:
            logger.exception("arXiv 搜索异常: {}", exc)
            raise RuntimeError(f"arXiv 搜索失败: {exc!s}") from exc

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

        sort_by = str(kwargs.get("sort_by", "relevance"))

        papers = await self.search(query, max_results, sort_by)
        return json.dumps(
            [p.to_dict() for p in papers],
            ensure_ascii=False,
        )
