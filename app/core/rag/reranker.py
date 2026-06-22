from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from app.models.schemas import RetrievalResult

class Reranker:

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
    ) -> None:
\
\
\

        self._model_name = model_name
        self._device = device
        self._model: Any = None

    def _load_model(self) -> Any:

        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise RuntimeError("请安装 sentence-transformers 以使用 Reranker") from e
        kwargs: dict[str, Any] = {}
        if self._device:
            kwargs["device"] = self._device
        self._model = CrossEncoder(self._model_name, **kwargs)
        return self._model

    async def rerank(
        self,
        query: str,
        documents: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:

        if not documents:
            return []
        if top_k <= 0:
            return []

        def _sync_predict() -> list[float]:
            model = self._load_model()
            pairs = [(query, d.content) for d in documents]
            raw = model.predict(pairs)
            if hasattr(raw, "tolist"):
                return list(raw.tolist())                               
            return list(raw)

        try:
            scores = await asyncio.to_thread(_sync_predict)
        except Exception as e:
            logger.exception("CrossEncoder 推理失败: {}", e)
            raise RuntimeError(f"重排序失败: {e}") from e

        if len(scores) != len(documents):
            raise RuntimeError("重排序分数数量与文档数量不一致")

        ranked = sorted(
            zip(documents, scores, strict=True),
            key=lambda x: float(x[1]),
            reverse=True,
        )
        out: list[RetrievalResult] = []
        for doc, sc in ranked[:top_k]:
            new_doc = doc.model_copy(deep=True)
            new_doc.score = float(sc)
            new_doc.metadata = {**new_doc.metadata, "rerank_score": float(sc)}
            out.append(new_doc)
        return out
