from __future__ import annotations

import asyncio
import math
import re
from collections import defaultdict
from typing import Any, Protocol, runtime_checkable

from loguru import logger

from app.models.enums import RetrievalMode
from app.models.schemas import RetrievalResult

@runtime_checkable
class EmbeddingProtocol(Protocol):

    def embed_query(self, text: str) -> list[float]:
        ...

@runtime_checkable
class MilvusSearchable(Protocol):

    def search(
        self,
        data: list[list[float]],
        anns_field: str,
        param: dict[str, Any],
        limit: int,
        output_fields: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        ...

def _tokenize(text: str) -> list[str]:

    return [t.lower() for t in re.findall(r"[\w\u4e00-\u9fff]+", text) if t]

class _BM25Index:

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_ids: list[str] = []
        self._documents: list[str] = []
        self._doc_freqs: list[dict[str, int]] = []
        self._doc_lens: list[int] = []
        self._avgdl: float = 0.0
        self._df: defaultdict[str, int] = defaultdict(int)
        self._N: int = 0
        self._idf: dict[str, float] = {}

    def clear(self) -> None:

        self._doc_ids.clear()
        self._documents.clear()
        self._doc_freqs.clear()
        self._doc_lens.clear()
        self._avgdl = 0.0
        self._df.clear()
        self._N = 0
        self._idf.clear()

    def add_document(self, doc_id: str, text: str) -> None:

        tokens = _tokenize(text)
        tf: defaultdict[str, int] = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        for t in tf:
            self._df[t] += 1
        self._doc_ids.append(doc_id)
        self._documents.append(text)
        self._doc_freqs.append(dict(tf))
        self._doc_lens.append(len(tokens))
        self._N += 1
        dl_sum = sum(self._doc_lens)
        self._avgdl = dl_sum / self._N if self._N else 0.0

        self._idf = {}
        for term, df in self._df.items():

            self._idf[term] = math.log(1.0 + (self._N - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:

        if self._N == 0:
            return []
        q_terms = _tokenize(query)
        if not q_terms:
            return []
        scores: dict[str, float] = {}
        for i, doc_id in enumerate(self._doc_ids):
            tf = self._doc_freqs[i]
            dl = self._doc_lens[i]
            s = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                idf = self._idf.get(term, 0.0)
                f = tf[term]
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1.0))
                s += idf * (f * (self.k1 + 1)) / denom
            if s > 0:
                scores[doc_id] = s
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return ranked

class MultiRetriever:

    anns_field: str = "embedding"
    text_field: str = "text"
    id_field: str = "id"

    metric_param: dict[str, Any] = {"metric_type": "L2", "params": {"nprobe": 16}}

    def __init__(self, milvus_client: Any, embedding_model: Any) -> None:
\
\
\

        self._milvus = milvus_client
        self._embed = embedding_model
        self._bm25 = _BM25Index()

        self._id_to_text: dict[str, str] = {}
        self._rrf_k: int = 60

    def register_keyword_documents(self, id_to_text: dict[str, str]) -> None:
\
\
\
\

        self._bm25.clear()
        self._id_to_text.clear()
        for did, text in id_to_text.items():
            self._bm25.add_document(did, text)
            self._id_to_text[did] = text
        logger.info("BM25 索引已更新，文档数: {}", len(id_to_text))

    def add_keyword_documents(self, id_to_text: dict[str, str]) -> None:

        added = 0
        for did, text in id_to_text.items():
            if did in self._id_to_text:
                continue
            self._bm25.add_document(did, text)
            self._id_to_text[did] = text
            added += 1
        if added:
            logger.info("BM25 增量追加 {} 条，总数 {}", added, len(self._id_to_text))

    async def retrieve(
        self, query: str, top_k: int = 10, mode: str = "hybrid"
    ) -> list[RetrievalResult]:

        mode_norm = mode.lower().strip()
        try:
            rm = RetrievalMode(mode_norm)
        except ValueError:
            logger.warning("未知检索模式 {}，回退为 hybrid", mode)
            rm = RetrievalMode.HYBRID

        if rm == RetrievalMode.VECTOR:
            return await self.vector_search(query, top_k)
        if rm == RetrievalMode.KEYWORD:
            return await self.keyword_search(query, top_k)
        return await self.hybrid_search(query, top_k)

    async def vector_search(self, query: str, top_k: int) -> list[RetrievalResult]:

        if not isinstance(self._embed, EmbeddingProtocol):
            raise TypeError("embedding_model 需实现 embed_query 方法")

        def _sync_embed() -> list[float]:
            return self._embed.embed_query(query)

        try:
            vector = await asyncio.to_thread(_sync_embed)
        except Exception as e:
            logger.exception("向量编码失败: {}", e)
            raise RuntimeError(f"向量编码失败: {e}") from e

        if not vector:
            raise RuntimeError("嵌入向量为空")

        def _sync_search() -> list[RetrievalResult]:
            return self._search_milvus(vector, top_k)

        try:
            return await asyncio.to_thread(_sync_search)
        except Exception as e:
            logger.exception("Milvus 向量检索失败: {}", e)
            raise RuntimeError(f"向量检索失败: {e}") from e

    def _search_milvus(self, vector: list[float], top_k: int) -> list[RetrievalResult]:

        coll = self._milvus
        if not isinstance(coll, MilvusSearchable):
            raise TypeError("milvus_client 需实现 search 方法")

        output_fields = [self.text_field, self.id_field]

        raw = coll.search(
            data=[vector],
            anns_field=self.anns_field,
            param=self.metric_param,
            limit=top_k,
            output_fields=output_fields,
        )
        results: list[RetrievalResult] = []
        hits = raw[0] if raw else []
        for rank, hit in enumerate(hits):
            entity = getattr(hit, "entity", None) or {}
            if hasattr(hit, "entity"):
                try:
                    entity = hit.entity.to_dict()                            
                except Exception:
                    entity = getattr(hit, "entity", {})

            score = float(getattr(hit, "distance", 0.0) or 0.0)

            text = entity.get(self.text_field) or entity.get("text") or ""
            rid = str(entity.get(self.id_field) or entity.get("id") or hit.id)
            results.append(
                RetrievalResult(
                    id=rid,
                    content=str(text),
                    score=score,
                    metadata={"rank": rank},
                    source="vector",
                )
            )
        return results

    async def keyword_search(self, query: str, top_k: int) -> list[RetrievalResult]:

        def _run() -> list[RetrievalResult]:
            ranked = self._bm25.search(query, top_k)
            out: list[RetrievalResult] = []
            for doc_id, sc in ranked:
                text = self._id_to_text.get(doc_id, "")
                out.append(
                    RetrievalResult(
                        id=doc_id,
                        content=text,
                        score=float(sc),
                        metadata={},
                        source="keyword",
                    )
                )
            return out

        try:
            return await asyncio.to_thread(_run)
        except Exception as e:
            logger.exception("BM25 检索失败: {}", e)
            raise RuntimeError(f"BM25 检索失败: {e}") from e

    def _rrf_fuse(
        self,
        lists: list[list[RetrievalResult]],
        top_k: int,
    ) -> list[RetrievalResult]:

        scores: dict[str, float] = {}
        id_best: dict[str, RetrievalResult] = {}
        k = self._rrf_k
        for lst in lists:
            for rank, item in enumerate(lst):
                rid = item.id
                scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank + 1)
                if rid not in id_best or item.score > id_best[rid].score:
                    id_best[rid] = item
        merged_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]
        merged: list[RetrievalResult] = []
        for rid in merged_ids:
            base = id_best[rid].model_copy(deep=True)
            base.score = scores[rid]
            base.metadata = {**base.metadata, "rrf": scores[rid]}
            base.source = "hybrid"
            merged.append(base)
        return merged

    async def hybrid_search(self, query: str, top_k: int) -> list[RetrievalResult]:

        vec_task = self.vector_search(query, top_k)
        kw_task = self.keyword_search(query, top_k)
        vec_res, kw_res = await asyncio.gather(vec_task, kw_task, return_exceptions=True)

        lists: list[list[RetrievalResult]] = []
        if isinstance(vec_res, list):
            lists.append(vec_res)
        else:
            logger.error("向量检索分支失败: {}", vec_res)
        if isinstance(kw_res, list) and kw_res:
            lists.append(kw_res)
        elif isinstance(kw_res, Exception):
            logger.error("关键词检索分支失败: {}", kw_res)

        if not lists:
            raise RuntimeError("混合检索两路均失败或无结果")
        if len(lists) == 1:
            return lists[0][:top_k]
        return self._rrf_fuse(lists, top_k)
