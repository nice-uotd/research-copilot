from __future__ import annotations
import asyncio
import json
import re
from typing import Any, Optional
import numpy as np
from loguru import logger
try:
    import redis.asyncio as aioredis
except ImportError:                    
    aioredis = None                            
_SEMANTIC_INDEX_KEY = "semantic_cache:index"
_SEMANTIC_ENTRY_PREFIX = "semantic_cache:entry:"
def _normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text
def _token_jaccard(a: str, b: str) -> float:
    sa = set(a.split()) if a else set()
    sb = set(b.split()) if b else set()
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0
class RedisCache:
    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        semantic_embedder: Any | None = None,
        max_semantic_scan: int = 200,
    ) -> None:
        if aioredis is None:
            raise RuntimeError("未安装 redis 包，请安装 redis>=5")
        self._url = url
        self._client: Any = aioredis.from_url(url, decode_responses=True)
        self._semantic_embedder = semantic_embedder
        self._max_semantic_scan = max(1, max_semantic_scan)
        self._embed_lock = asyncio.Lock()
    async def get(self, key: str) -> Optional[str]:
        try:
            val = await self._client.get(key)
            return val
        except Exception as exc:
            logger.exception("Redis GET 失败 key={}: {}", key, exc)
            raise
    async def set(self, key: str, value: str, ttl: int = 3600) -> None:
        try:
            if ttl > 0:
                await self._client.setex(key, ttl, value)
            else:
                await self._client.set(key, value)
        except Exception as exc:
            logger.exception("Redis SET 失败 key={}: {}", key, exc)
            raise
    async def _encode_query(self, query: str) -> np.ndarray:
        q = _normalize_text(query)
        if self._semantic_embedder is not None:
            def _run():
                emb = self._semantic_embedder.encode(
                    [q],
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                v = np.asarray(emb[0], dtype=np.float64)
                n = np.linalg.norm(v)
                if n > 0:
                    v = v / n
                return v
            async with self._embed_lock:
                return await asyncio.get_event_loop().run_in_executor(None, _run)
        dim = 256
        vec = np.zeros(dim, dtype=np.float64)
        for i in range(len(q) - 1):
            idx = (ord(q[i]) * 31 + ord(q[i + 1])) % dim
            vec[idx] += 1.0
        n = np.linalg.norm(vec)
        if n > 0:
            vec = vec / n
        return vec
    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 0:
            return 0.0
        return float(np.dot(a, b) / denom)
    async def semantic_get(self, query: str, threshold: float = 0.95) -> Optional[str]:
        try:
            q_vec = await self._encode_query(query)
            q_norm = _normalize_text(query)
            index_raw = await self._client.get(_SEMANTIC_INDEX_KEY)
            if not index_raw:
                return None
            try:
                entry_ids: list[str] = json.loads(index_raw)
            except json.JSONDecodeError:
                logger.warning("语义索引损坏，已忽略")
                return None
            best_sim = -1.0
            best_value: Optional[str] = None
            scanned = 0
            for eid in entry_ids:
                if scanned >= self._max_semantic_scan:
                    break
                scanned += 1
                raw = await self._client.get(_SEMANTIC_ENTRY_PREFIX + eid)
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                stored_query = str(payload.get("query_text", ""))
                value = str(payload.get("value", ""))
                emb_list = payload.get("embedding")
                if emb_list is not None:
                    s_vec = np.asarray(emb_list, dtype=np.float64)
                    sim = self._cosine(q_vec, s_vec)
                else:
                    sim = _token_jaccard(q_norm, _normalize_text(stored_query))
                if emb_list is None:
                    sim = max(sim, _token_jaccard(q_norm, _normalize_text(stored_query)))
                if sim > best_sim:
                    best_sim = sim
                    best_value = value
            if best_sim >= threshold and best_value is not None:
                logger.debug(
                    "语义缓存命中 similarity={:.4f} threshold={}",
                    best_sim,
                    threshold,
                )
                return best_value
            return None
        except Exception as exc:
            logger.exception("semantic_get 失败: {}", exc)
            raise
    async def semantic_set(
        self,
        query: str,
        value: str,
        ttl: int = 3600,
    ) -> None:
        q_vec = await self._encode_query(query)
        eid = str(abs(hash(_normalize_text(query))) % (10**12))
        payload = {
            "query_text": query,
            "value": value,
            "embedding": q_vec.tolist(),
        }
        key = _SEMANTIC_ENTRY_PREFIX + eid
        try:
            raw = json.dumps(payload, ensure_ascii=False)
            if ttl > 0:
                await self._client.setex(key, ttl, raw)
            else:
                await self._client.set(key, raw)
            index_raw = await self._client.get(_SEMANTIC_INDEX_KEY)
            ids: list[str] = []
            if index_raw:
                try:
                    ids = json.loads(index_raw)
                except json.JSONDecodeError:
                    ids = []
            if eid not in ids:
                ids.append(eid)
            await self._client.set(_SEMANTIC_INDEX_KEY, json.dumps(ids))
        except Exception as exc:
            logger.exception("semantic_set 失败: {}", exc)
            raise
    async def aclose(self) -> None:
        try:
            await self._client.aclose()
        except Exception as exc:
            logger.warning("关闭 Redis 连接时异常: {}", exc)
