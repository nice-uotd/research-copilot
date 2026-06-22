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
from app.core.tools.builtin.arxiv_search import ArxivSearchTool
from app.core.tools.builtin.calculator import CalculatorTool
from app.core.tools.builtin.scholar_search import ScholarSearchTool
from app.core.tools.builtin.search import WebSearchTool
from app.infrastructure.trace.tracer import Tracer
router = APIRouter(tags=["agent"])
_tracer = Tracer()
_agent: FunctionCallingAgent | None = None
_related_work_agent: FunctionCallingAgent | None = None
class RagSearchTool(BaseTool):
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
        tools = [
            RagSearchTool(rag),
            ws,
            CalculatorTool(),
            ArxivSearchTool(),
            ScholarSearchTool(),
        ]
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
_RELATED_WORK_SYSTEM = """\
你是一个学术写作助手，专门帮助研究者生成 Related Work 段落。
当用户给你一个研究主题时，请严格按以下步骤执行：
1. 调用 arxiv_search 搜索相关论文（建议 max_results=15）
2. 调用 scholar_search 搜索相关论文（建议 max_results=10），获取引用量数据
3. 从两个来源的结果中，筛选 8-12 篇最相关的高质量论文：
   - 优先选择引用量高的（>50 citations 为强信号）
   - 优先选择近 3 年的论文（体现 state-of-the-art）
   - 确保覆盖该主题的不同子方向
4. 生成一段 300-500 字的英文 Related Work 段落：
   - 使用 [Author et al., Year] 引用格式
   - 按研究子方向组织（不要逐篇罗列）
   - 指出各工作的贡献与局限
   - 末尾点明本工作与已有工作的区别
5. 在段落之后附上完整参考文献列表（按引用顺序编号）
重要：如果某个工具调用失败（如 API 限流），请用另一个工具的结果继续工作。
如果两个工具都失败了，请明确告知用户"当前 API 限流，请稍后重试"，不要编造论文。
输出格式：
## Related Work
[生成的段落]
## References
[1] Author1 et al. "Title." Venue, Year.
[2] ...
"""
def _get_related_work_agent() -> FunctionCallingAgent:
    global _related_work_agent
    if _related_work_agent is None:
        s = get_settings()
        if not s.openai_api_key:
            raise HTTPException(status_code=503, detail="未配置 OPENAI_API_KEY")
        client = AsyncOpenAI(
            api_key=s.openai_api_key,
            base_url=s.openai_api_base or None,
        )
        tools = [ArxivSearchTool(), ScholarSearchTool()]
        _related_work_agent = FunctionCallingAgent(
            llm_client=client,
            model=s.openai_model,
            tools=tools,
            max_iters=6,                                     
            system_prompt=_RELATED_WORK_SYSTEM,
        )
        logger.info("RelatedWorkAgent 已初始化")
    return _related_work_agent
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
class RelatedWorkRequest(BaseModel):
    topic: str = Field(min_length=1, description="研究主题或问题（英文效果更好）")
    max_papers: int = Field(default=12, ge=5, le=20, description="参考论文数量上限")
class RelatedWorkResponse(BaseModel):
    related_work: str
    steps: list[StepDTO]
    iterations: int
    finished: bool
    total_tokens: int
    trace_id: str
    papers_found: int
@router.post("/chat-related-work", response_model=RelatedWorkResponse)
async def chat_related_work(req: RelatedWorkRequest) -> RelatedWorkResponse:
    trace_id = str(uuid.uuid4())
    span = _tracer.start_trace(trace_id, "related_work")
    agent = _get_related_work_agent()
    query = (
        f"请为以下研究主题生成 Related Work 段落：\n\n"
        f"主题：{req.topic}\n\n"
        f"要求：筛选最相关的 {req.max_papers} 篇论文，生成英文 Related Work。"
    )
    try:
        result = await agent.run(query, max_iters=6)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("/chat-related-work 失败: {}", exc)
        _tracer.end_span(span, error=str(exc))
        raise HTTPException(
            status_code=500, detail=f"Related Work 生成失败: {exc!s}"
        ) from exc
    papers_found = 0
    for s in result.steps:
        if s.tool_name in ("arxiv_search", "scholar_search"):
            try:
                data = json.loads(s.tool_result)
                if isinstance(data, list):
                    papers_found += len(data)
            except (json.JSONDecodeError, TypeError):
                pass
    _tracer.end_span(
        span,
        result={
            "iterations": result.iterations,
            "tokens": result.total_tokens,
            "finished": result.finished,
            "papers_found": papers_found,
            "tools_used": [s.tool_name for s in result.steps],
        },
    )
    return RelatedWorkResponse(
        related_work=result.answer,
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
        papers_found=papers_found,
    )
