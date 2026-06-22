from __future__ import annotations
import uuid
from typing import Any
from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field
from app.config import get_settings
from app.core.tools.builtin.search import WebSearchResult, WebSearchTool
from app.infrastructure.llm.model_router import ModelConfig, ModelRouter
from app.infrastructure.trace.tracer import Tracer
router = APIRouter(tags=["web"])
_tracer = Tracer()
_tool: WebSearchTool | None = None
def _get_tool() -> WebSearchTool:
    global _tool
    if _tool is None:
        s = get_settings()
        _tool = WebSearchTool(
            tavily_api_key=s.tavily_api_key,
            max_results=s.web_search_max_results,
            timeout=s.web_search_timeout,
        )
        logger.info(
            "WebSearchTool 已创建，主提供商={}",
            "tavily" if _tool.has_tavily else "ddgs",
        )
    return _tool
class WebSearchRequest(BaseModel):
    query: str = Field(min_length=1, description="搜索查询")
    max_results: int = Field(default=5, ge=1, le=10)
class WebResultDTO(BaseModel):
    title: str
    url: str
    content: str
    score: float
    source: str
class WebSearchResponse(BaseModel):
    query: str
    provider: str
    results: list[WebResultDTO]
class ChatWebRequest(BaseModel):
    query: str = Field(min_length=1, description="用户问题")
    max_results: int = Field(default=5, ge=1, le=10)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
class ChatWebResponse(BaseModel):
    answer: str
    contexts: list[WebResultDTO]
    model: str
    trace_id: str
    provider: str
    usage: dict[str, Any] | None = None
_SYSTEM_PROMPT = (
    "你是联网研究助手。仅根据提供的「网页摘要」作答；"
    "回答中引用具体观点时使用 [1]、[2] 等编号，编号与下方摘要一一对应。"
    "若摘要不足以回答，明确说明并建议追加搜索词。回答用中文，专业术语保留原文。"
)
def _build_router() -> ModelRouter:
    s = get_settings()
    if not s.openai_api_key:
        raise HTTPException(status_code=503, detail="未配置 OPENAI_API_KEY")
    return ModelRouter(
        [
            ModelConfig(
                model_id=s.openai_model,
                api_key=s.openai_api_key,
                base_url=s.openai_api_base or None,
                priority=0,
                weight=1.0,
            )
        ]
    )
def _format_contexts(results: list[WebSearchResult]) -> str:
    if not results:
        return "（无搜索结果）"
    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(
            f"[{i}] {r.title}\n  URL: {r.url}\n  摘要: {r.content[:500]}"
        )
    return "\n\n".join(lines)
@router.post("/web-search", response_model=WebSearchResponse)
async def web_search(req: WebSearchRequest) -> WebSearchResponse:
    tool = _get_tool()
    try:
        results = await tool.search(req.query, req.max_results)
    except Exception as exc:
        logger.exception("/web-search 失败: {}", exc)
        raise HTTPException(status_code=500, detail=f"搜索失败: {exc!s}") from exc
    provider = (
        results[0].source
        if results
        else ("tavily" if tool.has_tavily else "ddgs")
    )
    return WebSearchResponse(
        query=req.query,
        provider=provider,
        results=[WebResultDTO(**r.to_dict()) for r in results],
    )
@router.post("/chat-web", response_model=ChatWebResponse)
async def chat_web(req: ChatWebRequest) -> ChatWebResponse:
    trace_id = str(uuid.uuid4())
    span = _tracer.start_trace(trace_id, "chat_web")
    tool = _get_tool()
    try:
        contexts = await tool.search(req.query, req.max_results)
    except Exception as exc:
        logger.exception("web search 失败: {}", exc)
        _tracer.end_span(span, error=str(exc))
        raise HTTPException(status_code=500, detail=f"搜索失败: {exc!s}") from exc
    user_prompt = (
        f"## 网页摘要\n{_format_contexts(contexts)}\n\n"
        f"## 用户问题\n{req.query}\n\n"
        "请基于上述摘要作答，必要时使用 [n] 引用对应来源。"
    )
    router_llm = _build_router()
    try:
        resp = await router_llm.chat(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=req.temperature,
        )
    except Exception as exc:
        logger.exception("chat-web 生成失败: {}", exc)
        _tracer.end_span(span, error=str(exc))
        raise HTTPException(status_code=500, detail=f"生成失败: {exc!s}") from exc
    provider = contexts[0].source if contexts else "none"
    _tracer.end_span(
        span,
        result={
            "model": resp.model_id,
            "ctx_count": len(contexts),
            "usage": resp.usage,
            "provider": provider,
        },
    )
    return ChatWebResponse(
        answer=resp.content,
        contexts=[WebResultDTO(**r.to_dict()) for r in contexts],
        model=resp.model_id,
        trace_id=trace_id,
        provider=provider,
        usage=resp.usage,
    )
