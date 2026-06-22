from __future__ import annotations

import re
from typing import Any, Protocol, runtime_checkable

from loguru import logger

from app.models.schemas import Citation, Message, RAGResponse, RetrievalResult

@runtime_checkable
class RAGLLMProtocol(Protocol):

    async def ainvoke(self, input: Any, **kwargs: Any) -> Any:
        ...

class RAGGenerator:

    def __init__(
        self,
        llm: Any,
        system_prompt: str | None = None,
        model_name: str | None = None,
    ) -> None:
\
\
\
\

        self._llm = llm
        self._system_prompt = system_prompt or (

        )
        self._model_name = model_name

    def _build_messages(
        self,
        query: str,
        contexts: list[RetrievalResult],
        chat_history: list[Any],
    ) -> list[dict[str, str]]:

        ctx_lines = []
        for i, c in enumerate(contexts, start=1):
            ctx_lines.append(f"[{i}] (id={c.id})\n{c.content}")
        context_block = "\n\n".join(ctx_lines) if ctx_lines else "（无检索上下文）"

        history_lines: list[str] = []
        for msg in chat_history:
            if isinstance(msg, Message):
                history_lines.append(f"{msg.role.value}: {msg.content}")
            elif isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                history_lines.append(f"{role}: {content}")
            else:
                history_lines.append(str(msg))

        user_content = (

        )

        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _extract_citations(self, answer: str, contexts: list[RetrievalResult]) -> list[Citation]:

        refs = [int(x) for x in re.findall(r"\[(\d+)\]", answer)]
        citations: list[Citation] = []
        seen: set[int] = set()
        for idx in refs:
            if idx in seen or idx < 1 or idx > len(contexts):
                continue
            seen.add(idx)
            r = contexts[idx - 1]
            snippet = r.content[:200] + ("..." if len(r.content) > 200 else "")
            citations.append(
                Citation(index=idx, result_id=r.id, snippet=snippet),
            )
        return citations

    def _parse_llm_output(self, raw: Any) -> str:

        if raw is None:
            return ""
        if isinstance(raw, str):
            return raw
        if hasattr(raw, "content"):
            return str(getattr(raw, "content", ""))
        if isinstance(raw, dict) and "content" in raw:
            return str(raw["content"])
        return str(raw)

    async def generate(
        self,
        query: str,
        contexts: list[RetrievalResult],
        chat_history: list[Any],
    ) -> RAGResponse:

        messages = self._build_messages(query, contexts, chat_history)

        try:
            if isinstance(self._llm, RAGLLMProtocol):
                raw = await self._llm.ainvoke(messages)
            elif callable(getattr(self._llm, "ainvoke", None)):
                raw = await self._llm.ainvoke(messages)                      
            else:
                raise TypeError("llm 需实现异步 ainvoke(messages)")
        except Exception as e:
            logger.exception("RAG 生成调用失败: {}", e)
            raise RuntimeError(f"生成失败: {e}") from e

        answer = self._parse_llm_output(raw).strip()
        citations = self._extract_citations(answer, contexts)

        return RAGResponse(
            answer=answer,
            citations=citations,
            raw_contexts=list(contexts),
            model=self._model_name,
        )
