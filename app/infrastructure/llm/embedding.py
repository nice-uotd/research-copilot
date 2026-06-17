# -*- coding: utf-8 -*-
"""Embedding 客户端：OpenAI 兼容协议（默认对接阿里百炼 text-embedding-v3）。"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger
from openai import AsyncOpenAI


class EmbeddingClient:
    """同步 + 异步嵌入接口；与 LangChain Embeddings 鸭子兼容（embed_query / embed_documents）。"""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "text-embedding-v3",
        dim: int = 1024,
        batch_size: int = 10,
    ) -> None:
        self._model = model
        self._dim = dim
        self._batch_size = max(1, batch_size)
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**kwargs)

    @property
    def dim(self) -> int:
        return self._dim

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """异步批量嵌入。"""
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            try:
                resp = await self._client.embeddings.create(model=self._model, input=batch)
                out.extend([d.embedding for d in resp.data])
            except Exception as exc:
                logger.exception("嵌入调用失败 batch_start={}: {}", i, exc)
                raise
        return out

    def embed_query(self, text: str) -> list[float]:
        """同步单条嵌入（兼容 LangChain Embeddings 接口）。"""
        return asyncio.run(self.aembed([text]))[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """同步批量嵌入。"""
        return asyncio.run(self.aembed(texts))
