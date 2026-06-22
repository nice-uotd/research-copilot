from __future__ import annotations
import re
from typing import List
from loguru import logger
from app.core.tools.base import BaseTool
class ToolRouter:
    def __init__(self, max_tools: int = 5) -> None:
        self._max_tools = max(1, max_tools)
    async def route(self, query: str, available_tools: List[BaseTool]) -> List[BaseTool]:
        if not available_tools:
            return []
        q = query.strip().lower()
        scored: list[tuple[float, BaseTool]] = []
        for tool in available_tools:
            score = 0.0
            if tool.name.lower() in q:
                score += 3.0
            for token in re.split(r"\W+", tool.description.lower()):
                if len(token) > 1 and token in q:
                    score += 1.0
            if any(k in q for k in ("搜", "查", "新闻", "search", "web")) and "search" in tool.name:
                score += 2.0
            if any(k in q for k in ("算", "计算", "calc", "+", "-", "*", "/")):
                if "calc" in tool.name or "calculator" in tool.name:
                    score += 2.0
            if any(k in q for k in ("sql", "数据库", "查询表", "database")):
                if "database" in tool.name or "db" in tool.name:
                    score += 2.0
            scored.append((score, tool))
        scored.sort(key=lambda x: -x[0])
        picked = [t for _, t in scored[: self._max_tools]]
        if all(s == 0.0 for s, _ in scored):
            picked = available_tools[: self._max_tools]
        logger.debug(
            "工具路由 query_preview={} 选中: {}",
            q[:80],
            [t.name for t in picked],
        )
        return picked
