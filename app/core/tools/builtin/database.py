from __future__ import annotations

from typing import Any

from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.tools.base import BaseTool, ToolParameter

class DatabaseQueryTool(BaseTool):

    def __init__(self, session_factory: Any | None = None) -> None:
\
\
\

        super().__init__()
        self.name = "database_query"
        self.description = "在只读模式下执行 SQL 查询并返回行列表（禁止写操作）。"
        self.parameters = [
            ToolParameter(
                name="sql",
                type="string",
                description="只读 SQL，必须以 SELECT 开头",
                required=True,
            )
        ]
        self._session_factory = session_factory

    def _validate_sql(self, sql: str) -> str:
        s = sql.strip().rstrip(";")
        lower = s.lower()
        if not lower.startswith("select"):
            raise ValueError("仅允许 SELECT 查询")
        forbidden = ("insert", "update", "delete", "drop", "alter", "truncate", "create")
        for bad in forbidden:
            if bad in lower:
                raise ValueError(f"查询包含禁止关键字: {bad}")
        return s

    async def execute(self, **kwargs: Any) -> Any:

        sql = str(kwargs.get("sql", "")).strip()
        if not sql:
            raise ValueError("参数 sql 不能为空")
        safe_sql = self._validate_sql(sql)

        session: AsyncSession | None = kwargs.get("session")
        if session is None and self._session_factory is None:
            raise RuntimeError("未提供 session 且未配置 session_factory")

        async def _run(sess: AsyncSession) -> list[dict[str, Any]]:
            result = await sess.execute(text(safe_sql))
            rows = result.mappings().all()

            return [dict(r) for r in rows]

        if session is not None:
            out = await _run(session)
            logger.info("database_query 返回 {} 行", len(out))
            return out

        async with self._session_factory() as sess:                      
            out = await _run(sess)
            logger.info("database_query 返回 {} 行", len(out))
            return out
