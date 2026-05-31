# -*- coding: utf-8 -*-
"""Function Calling Agent：基于 OpenAI tools 协议的工具循环。

设计：
  - 每轮让 LLM 决定调哪些工具（可并行多调）；
  - 执行后把 observation 灌回上下文，进入下一轮；
  - 直到 LLM 不再返回 tool_calls（说明它要给最终答案）或达到 max_iters；
  - 全程记录 step 轨迹用于面试演示与 trace 追踪。

为什么不用 react_agent.py：
  文本解析式 ReAct 依赖 LLM 严格按 "Thought:/Action:" 输出，鲁棒性差；
  function calling 是当前生产标准（OpenAI/Anthropic/DeepSeek/Qwen 全部原生支持）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from app.core.tools.base import BaseTool


@dataclass
class AgentStep:
    """单步 Agent 工具调用记录。"""

    step: int
    tool_name: str
    tool_args: dict[str, Any]
    tool_result: str  # 已截断到 1500 char
    error: str | None = None


@dataclass
class AgentTrace:
    """完整 Agent 运行轨迹。"""

    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    total_tokens: int = 0
    iterations: int = 0
    finished: bool = True


class FunctionCallingAgent:
    """基于 OpenAI function calling 的 Agent 循环。"""

    def __init__(
        self,
        llm_client: AsyncOpenAI,
        model: str,
        tools: list[BaseTool],
        max_iters: int = 5,
        system_prompt: str | None = None,
    ) -> None:
        self._client = llm_client
        self._model = model
        self._tools = {t.name: t for t in tools}
        self.max_iters = max(1, max_iters)
        self._system_prompt = system_prompt or self._default_system()

    def _default_system(self) -> str:
        names = list(self._tools.keys())
        return (
            f"你是一个研究助手 Agent，可调用工具：{names}。\n"
            "选择准则：\n"
            "- 项目内部文档/技术细节问题（混合检索、重排、Function Calling 设计、"
            "熔断、评测方法等）→ 调用 rag_search\n"
            "- 时效性、外部技术、行业动态 → 调用 web_search\n"
            "- 涉及具体数值计算 → 调用 calculator\n"
            "- 不需要工具时直接回答；多个工具结果可串行调用，最终综合回答\n"
            "回答时引用工具结果中的关键信息，用中文。"
        )

    def _build_tools_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.schema_parameters(),
                },
            }
            for t in self._tools.values()
        ]

    async def _execute_tool(
        self, name: str, args: dict[str, Any]
    ) -> tuple[str, str | None]:
        """执行单个工具调用，返回 (result_str, error_or_none)。"""
        tool = self._tools.get(name)
        if tool is None:
            msg = f"错误：工具 [{name}] 不存在，可用工具={list(self._tools)}"
            return msg, "tool_not_found"
        try:
            raw = await tool.execute(**args)
            if isinstance(raw, str):
                return raw, None
            return json.dumps(raw, ensure_ascii=False, default=str), None
        except Exception as exc:  # noqa: BLE001
            logger.exception("工具 [{}] 执行失败 args={}", name, args)
            return f"工具执行错误: {exc!s}", str(exc)

    async def run(self, query: str, max_iters: int | None = None) -> AgentTrace:
        """执行 Agent 循环。"""
        max_iters = max(1, max_iters or self.max_iters)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": query},
        ]
        tools_schema = self._build_tools_schema()
        steps: list[AgentStep] = []
        total_tokens = 0
        step_idx = 0

        for it in range(max_iters):
            try:
                resp = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=tools_schema,
                    tool_choice="auto",
                    temperature=0.2,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("LLM 调用失败 iter={}: {}", it, exc)
                return AgentTrace(
                    answer=f"Agent 失败：{exc!s}",
                    steps=steps,
                    total_tokens=total_tokens,
                    iterations=it,
                    finished=False,
                )

            if resp.usage and resp.usage.total_tokens:
                total_tokens += resp.usage.total_tokens

            msg = resp.choices[0].message
            tool_calls = msg.tool_calls or []

            if not tool_calls:
                # LLM 不再要求工具 → 最终答案
                return AgentTrace(
                    answer=msg.content or "",
                    steps=steps,
                    total_tokens=total_tokens,
                    iterations=it + 1,
                    finished=True,
                )

            # 把 assistant 的 tool_calls 消息回灌
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            # 执行每个工具调用并把 observation 追加为 tool 消息
            for tc in tool_calls:
                step_idx += 1
                try:
                    args = (
                        json.loads(tc.function.arguments)
                        if tc.function.arguments
                        else {}
                    )
                except json.JSONDecodeError:
                    args = {}
                result, err = await self._execute_tool(tc.function.name, args)
                steps.append(
                    AgentStep(
                        step=step_idx,
                        tool_name=tc.function.name,
                        tool_args=args,
                        tool_result=result[:1500],
                        error=err,
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result[:6000],
                    }
                )

        # 循环结束未给最终答案
        return AgentTrace(
            answer="（Agent 达到最大循环次数，未给出最终答案）",
            steps=steps,
            total_tokens=total_tokens,
            iterations=max_iters,
            finished=False,
        )
