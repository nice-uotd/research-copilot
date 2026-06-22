from __future__ import annotations
import json
import re
from typing import Any
from loguru import logger
from openai import AsyncOpenAI
from app.models.schemas import RetrievalResult
_JUDGE_SYSTEM = (
    "You are a precise relevance judge for a research assistant. "
    "Given a user query and numbered passages, score each passage 0-100: "
    "100=directly answers, 70-99=contains key info, 30-69=tangentially related, 0-29=irrelevant. "
    'Output ONLY a JSON array, no prose, no markdown: [{"index": int, "score": int}, ...]. '
    "Cover every candidate by 1-based index."
)
def _build_user_prompt(
    query: str,
    candidates: list[RetrievalResult],
    snippet_chars: int,
) -> str:
    parts = [f"Query: {query}", "", "Candidates:"]
    for i, c in enumerate(candidates, start=1):
        snippet = c.content.replace("\n", " ").strip()[:snippet_chars]
        parts.append(f"[{i}] {snippet}")
    parts.append("")
    parts.append('Return only: [{"index": int, "score": int}, ...]')
    return "\n".join(parts)
def _parse_scores(raw: str, n: int) -> dict[int, int] | None:
    if not raw:
        return None
    blobs = re.findall(r"\[[\s\S]*?\]", raw)
    for blob in blobs:
        try:
            arr = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(arr, list) or not arr:
            continue
        out: dict[int, int] = {}
        for item in arr:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            score = item.get("score")
            if (
                isinstance(idx, int)
                and 1 <= idx <= n
                and isinstance(score, (int, float))
            ):
                out[idx] = max(0, min(100, int(score)))
        if out:
            return out
    return None
class LLMJudge:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        temperature: float = 0.0,
        snippet_chars: int = 500,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = max(0.0, temperature)
        self._snippet_chars = max(100, snippet_chars)
    async def rerank(
        self,
        query: str,
        documents: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if not documents or top_k <= 0:
            return []
        if len(documents) == 1:
            return documents[:1]
        user_prompt = _build_user_prompt(query, documents, self._snippet_chars)
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self._temperature,
            )
        except Exception as exc:
            logger.warning("LLM judge 调用失败，回退原序: {}", exc)
            return documents[:top_k]
        raw = resp.choices[0].message.content if resp.choices else ""
        scores = _parse_scores(raw or "", len(documents))
        if not scores:
            logger.warning(
                "LLM judge 解析失败，回退原序; raw_preview={}",
                (raw or "")[:200],
            )
            return documents[:top_k]
        ranked_indices = sorted(scores.keys(), key=lambda i: -scores[i])
        out: list[RetrievalResult] = []
        seen_zero_indexed: set[int] = set()
        for idx in ranked_indices[:top_k]:
            doc = documents[idx - 1].model_copy(deep=True)
            score_norm = scores[idx] / 100.0
            doc.score = score_norm
            doc.metadata = {
                **doc.metadata,
                "llm_judge_score": score_norm,
                "llm_judged": True,
            }
            out.append(doc)
            seen_zero_indexed.add(idx - 1)
        for i, d in enumerate(documents):
            if len(out) >= top_k:
                break
            if i in seen_zero_indexed:
                continue
            out.append(d)
        return out
