from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.rag.service import get_rag_service
from app.infrastructure.llm.model_router import ModelConfig, ModelRouter
from app.infrastructure.trace.tracer import Tracer
from app.models.schemas import Citation, RetrievalResult

router = APIRouter(tags=["rag"])

_tracer = Tracer()

class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, description="检索问题")
    top_k: int = Field(default=5, ge=1, le=50)
    mode: str = Field(default="hybrid", description="vector | keyword | hybrid")
    use_rerank: bool | None = Field(
        default=None,
        description="True 强制开启 Cross-Encoder 重排，False 强制关闭，None 跟随全局开关",
    )
    use_llm_judge: bool | None = Field(
        default=None,
        description="True 启用 LLM-as-judge 终排，False 关闭，None 跟随全局",
    )

class RetrieveResponse(BaseModel):
    query: str
    mode: str
    reranked: bool
    llm_judged: bool
    results: list[RetrievalResult]

class ChatRAGRequest(BaseModel):
    query: str = Field(min_length=1, description="用户问题")
    top_k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="hybrid", description="vector | keyword | hybrid")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    use_rerank: bool | None = Field(default=None, description="是否启用 bge 重排")
    use_llm_judge: bool | None = Field(default=None, description="是否启用 LLM-as-judge 终排")

class ChatRAGResponse(BaseModel):
    answer: str
    citations: list[Citation]
    contexts: list[RetrievalResult]
    model: str
    trace_id: str
    usage: dict[str, Any] | None = None

_SYSTEM_PROMPT = (

)

def _build_router() -> ModelRouter:
    settings = get_settings()
    if not settings.openai_api_key:
        raise HTTPException(status_code=503, detail="未配置 OPENAI_API_KEY")
    return ModelRouter(
        [
            ModelConfig(
                model_id=settings.openai_model,
                api_key=settings.openai_api_key,
                base_url=settings.openai_api_base or None,
                priority=0,
                weight=1.0,
            )
        ]
    )

def _format_contexts(contexts: list[RetrievalResult]) -> str:
    lines = []
    for i, c in enumerate(contexts, start=1):
        lines.append(f"[{i}] (id={c.id[:8]} score={c.score:.4f})\n{c.content}")
    return "\n\n".join(lines) if lines else "（无检索结果）"

def _extract_citations(answer: str, contexts: list[RetrievalResult]) -> list[Citation]:
    import re

    refs = [int(x) for x in re.findall(r"\[(\d+)\]", answer)]
    seen: set[int] = set()
    citations: list[Citation] = []
    for idx in refs:
        if idx in seen or idx < 1 or idx > len(contexts):
            continue
        seen.add(idx)
        r = contexts[idx - 1]
        snippet = r.content[:200] + ("..." if len(r.content) > 200 else "")
        citations.append(Citation(index=idx, result_id=r.id, snippet=snippet))
    return citations

@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest) -> RetrieveResponse:

    rag = get_rag_service()
    try:
        results = await rag.retrieve(
            query=req.query,
            top_k=req.top_k,
            mode=req.mode,
            use_rerank=req.use_rerank,
            use_llm_judge=req.use_llm_judge,
        )
    except Exception as exc:
        logger.exception("/retrieve 失败: {}", exc)
        raise HTTPException(status_code=500, detail=f"检索失败: {exc!s}") from exc
    reranked = bool(results) and bool(results[0].metadata.get("reranked"))
    llm_judged = bool(results) and bool(results[0].metadata.get("llm_judged"))
    return RetrieveResponse(
        query=req.query,
        mode=req.mode,
        reranked=reranked,
        llm_judged=llm_judged,
        results=results,
    )

@router.post("/chat-rag", response_model=ChatRAGResponse)
async def chat_rag(req: ChatRAGRequest) -> ChatRAGResponse:

    trace_id = str(uuid.uuid4())
    span = _tracer.start_trace(trace_id, "chat_rag")

    rag = get_rag_service()
    try:
        contexts = await rag.retrieve(
            query=req.query,
            top_k=req.top_k,
            mode=req.mode,
            use_rerank=req.use_rerank,
            use_llm_judge=req.use_llm_judge,
        )
    except Exception as exc:
        logger.exception("RAG 检索失败: {}", exc)
        _tracer.end_span(span, error=str(exc))
        raise HTTPException(status_code=500, detail=f"检索失败: {exc!s}") from exc

    user_prompt = (

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
        logger.exception("RAG 生成失败: {}", exc)
        _tracer.end_span(span, error=str(exc))
        raise HTTPException(status_code=500, detail=f"生成失败: {exc!s}") from exc

    citations = _extract_citations(resp.content, contexts)
    _tracer.end_span(
        span,
        result={"model": resp.model_id, "ctx_count": len(contexts), "usage": resp.usage},
    )

    return ChatRAGResponse(
        answer=resp.content,
        citations=citations,
        contexts=contexts,
        model=resp.model_id,
        trace_id=trace_id,
        usage=resp.usage,
    )
