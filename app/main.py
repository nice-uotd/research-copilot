# -*- coding: utf-8 -*-
"""FastAPI 应用入口：生命周期内初始化数据库引擎、SQLite 表与 RAG 服务。"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from loguru import logger

from app.api.routes import chat, document, health
from app.api.routes import agent as agent_route
from app.api.routes import rag as rag_route
from app.api.routes import web as web_route
from app.config import get_settings
from app.core.rag.service import init_rag_service
from app.infrastructure.database.models import Base
from app.infrastructure.database.session import configure_session, init_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("启动 {} ({})", settings.app_name, settings.app_env)

    Path("./data").mkdir(parents=True, exist_ok=True)

    engine = init_engine(settings.database_url)
    factory = configure_session(engine)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("SQLite 表已就绪")

    rag = init_rag_service(settings)
    await rag.rebuild_bm25_from_db(factory)

    app.state.engine = engine
    app.state.rag = rag
    yield
    await engine.dispose()
    logger.info("关闭 {}", settings.app_name)


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        lifespan=lifespan,
    )
    application.include_router(health.router, prefix=settings.api_prefix)
    application.include_router(chat.router, prefix=settings.api_prefix)
    application.include_router(document.router, prefix=settings.api_prefix)
    application.include_router(rag_route.router, prefix=settings.api_prefix)
    application.include_router(web_route.router, prefix=settings.api_prefix)
    application.include_router(agent_route.router, prefix=settings.api_prefix)

    @application.get("/", include_in_schema=False)
    async def _root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    return application


app = create_app()
