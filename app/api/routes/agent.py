# -*- coding: utf-8 -*-
"""Agent API：function calling 驱动的多工具研究助手。"""

from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.agent.function_calling_agent import FunctionCallingAgent
from app.core.rag.service import get_rag_service
from app.core.tools.base import BaseTool, ToolParameter
from app.core.tools.builtin.calculator import CalculatorTool
from app.core.tools.builtin.search import WebSearchTool
from app.infrastructure.trace.tracer import Tracer

router = APIRouter(tags=["agent"])

_tracer = Tracer()
_agent: FunctionCallingAgent | None = None


class RagSearchTool(BaseTool):
    """把 RAG 服务包成 Agent 工具。"""

    def __init__(self, rag_service: Any) -> None:
        super().__init__()
        self.name = "rag_search"
        self.description = (
            "搜索内部知识库（项目文档、技术资料、Agent/RAG 设计说明等）。"
            "适用：询问混合检索、RRF、重排器、Function Calling、熔断、评测方法等"
            "项目设计与实现细节问题。"
            "返回 JSON 数组，每元素含 id、score、content（文档原文片段）。"
        )
        self.parameters = [
            ToolParameter(
                name="query",
                type="string",
                description="检索关键词或自然语言问题",
                required=True,
            ),
            ToolParameter(
                name="top_k",
                type="integer",
                description="返回片段数量（1-10），默认 3",
                required=False,
            ),
        ]
        self._rag = rag_service

    async def execute(self, **kwargs: Any) -> str:
        q = str(kwargs.get("query", "")).strip()
        if not q:
            raise ValueError("参数 query 不能为空")
        raw_k = kwargs.get("top_k", 3)
        try:
            top_k = max(1, min(int(raw_k), 10))
        except (TypeError, ValueError):
            top_k = 3
        results = await self._rag.retrieve(
            query=q, top_k=top_k, mode="hybrid", use_rerank=False
        )
        return json.dumps(
            [
                {
                    "id": r.id[:8],
                    "score": round(float(r.score), 4),
                    "content": r.content[:600],
                }
                for r in results
            ],
            ensure_ascii=False,
        )


def _get_agent() -> FunctionCallingAgent:
    """懒加载 Agent 单例。"""
    global _agent
    if _agent is None:
        s = get_settings()
        if not s.openai_api_key:
            raise HTTPException(status_code=503, detail="未配置 OPENAI_API_KEY")
        client = AsyncOpenAI(
            api_key=s.openai_api_key,
            base_url=s.openai_api_base or None,
        )
        rag = get_rag_service()
        ws = WebSearchTool(
            tavily_api_key=s.tavily_api_key,
            max_results=s.web_search_max_results,
            timeout=s.web_search_timeout,
        )
        tools = [RagSearchTool(rag), ws, CalculatorTool()]
        _agent = FunctionCallingAgent(
            llm_client=client,
            model=s.openai_model,
            tools=tools,
            max_iters=5,
        )
        logger.info(
            "FunctionCallingAgent 已初始化 tools={}",
            [t.name for t in tools],
        )
    return _agent


class ChatAgentRequest(BaseModel):
    query: str = Field(min_length=1, description="用户问题")
    max_iters: int = Field(default=5, ge=1, le=10)


class StepDTO(BaseModel):
    step: int
    tool_name: str
    tool_args: dict[str, Any]
    tool_result: str
    error: str | None = None


class ChatAgentResponse(BaseModel):
    answer: str
    steps: list[StepDTO]
    iterations: int
    finished: bool
    total_tokens: int
    trace_id: str


@router.post("/chat-agent", response_model=ChatAgentResponse)
async def chat_agent(req: ChatAgentRequest) -> ChatAgentResponse:
    """Agent 入口：LLM 自主选择 rag_search / web_search / calculator。"""
    trace_id = str(uuid.uuid4())
    span = _tracer.start_trace(trace_id, "chat_agent")
    agent = _get_agent()
    try:
        result = await agent.run(req.query, max_iters=req.max_iters)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("/chat-agent 失败: {}", exc)
        _tracer.end_span(span, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Agent 失败: {exc!s}") from exc

    _tracer.end_span(
        span,
        result={
            "iterations": result.iterations,
            "tokens": result.total_tokens,
            "finished": result.finished,
            "steps": len(result.steps),
            "tools_used": [s.tool_name for s in result.steps],
        },
    )

    return ChatAgentResponse(
        answer=result.answer,
        steps=[
            StepDTO(
                step=s.step,
                tool_name=s.tool_name,
                tool_args=s.tool_args,
                tool_result=s.tool_result,
                error=s.error,
            )
            for s in result.steps
        ],
        iterations=result.iterations,
        finished=result.finished,
        total_tokens=result.total_tokens,
        trace_id=trace_id,
    )
