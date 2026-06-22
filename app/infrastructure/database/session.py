from __future__ import annotations
import re
from collections.abc import AsyncGenerator
from typing import Any
from loguru import logger
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
_default_url = "postgresql+asyncpg://postgres:postgres@localhost:5432/agent_db"
def normalize_async_database_url(url: str) -> str:
    if "+asyncpg" in url:
        return url
    u = url.replace("postgresql+psycopg2://", "postgresql+asyncpg://")
    u = u.replace("postgres://", "postgresql+asyncpg://")
    u = re.sub(
        r"^postgresql://",
        "postgresql+asyncpg://",
        u,
    )
    return u
def init_engine(database_url: str | None = None, **engine_kwargs: Any) -> AsyncEngine:
    url = normalize_async_database_url(database_url or _default_url)
    kwargs = {"echo": False, "pool_pre_ping": True}
    kwargs.update(engine_kwargs)
    engine = create_async_engine(url, **kwargs)
    logger.info("数据库引擎已初始化（已隐藏凭据）")
    return engine
_engine: AsyncEngine | None = None
async_session_factory: async_sessionmaker[AsyncSession] | None = None
def configure_session(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    global _engine, async_session_factory
    _engine = engine
    async_session_factory = async_sessionmaker(
        engine,
        expire_on_commit=False,
        autoflush=False,
    )
    return async_session_factory
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    if async_session_factory is None:
        raise RuntimeError("请先调用 configure_session(init_engine(...))")
    async with async_session_factory() as session:
        yield session
