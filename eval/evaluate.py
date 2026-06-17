# -*- coding: utf-8 -*-
"""检索评测脚本：HitRate@K + MRR，逐 mode 横向对比。

用法：
  .venv/bin/python eval/evaluate.py
  .venv/bin/python eval/evaluate.py --qa eval/qa_set.json --modes hybrid vector keyword
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx


def first_hit_rank(results: list[dict], expected_substrings: list[str]) -> int | None:
    """返回首个命中（chunk content 包含任意 expected substring）的 1-索引名次；未命中返回 None。"""
    if not expected_substrings:
        return None
    needles = [s.lower() for s in expected_substrings if s]
    for i, r in enumerate(results, start=1):
        content = (r.get("content") or "").lower()
        if any(n in content for n in needles):
            return i
    return None


async def run_one(
    client: httpx.AsyncClient,
    base_url: str,
    question: str,
    mode: str,
    top_k: int,
    use_rerank: bool | None = None,
    use_llm_judge: bool | None = None,
) -> dict:
    payload: dict = {"query": question, "top_k": top_k, "mode": mode}
    if use_rerank is not None:
        payload["use_rerank"] = use_rerank
    if use_llm_judge is not None:
        payload["use_llm_judge"] = use_llm_judge
    try:
        r = await client.post(f"{base_url}/api/v1/retrieve", json=payload, timeout=180.0)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e), "results": []}


async def evaluate(
    qa_path: str,
    base_url: str,
    top_k: int,
    modes: list[str],
    out_path: str,
) -> int:
    """modes 元素格式（可叠加后缀，从右到左解析）：
      - mode 名（如 hybrid）→ 跟全局
      - +rerank / -rerank → 强制 bge 开/关
      - +judge / -judge → 强制 LLM judge 开/关
      例：hybrid+rerank+judge / hybrid-rerank+judge / vector-rerank-judge
    """
    items = json.loads(Path(qa_path).read_text(encoding="utf-8"))["items"]
    if not items:
        print("评测集为空", file=sys.stderr)
        return 1

    def _parse(label: str) -> tuple[str, bool | None, bool | None]:
        rerank: bool | None = None
        judge: bool | None = None
        rest = label
        # 尾部反复抽 +foo/-foo 后缀
        while True:
            stripped = False
            for suffix, val in (
                ("+rerank", True), ("-rerank", False),
                ("+judge", True), ("-judge", False),
            ):
                if rest.endswith(suffix):
                    if "rerank" in suffix:
                        rerank = val
                    else:
                        judge = val
                    rest = rest[: -len(suffix)]
                    stripped = True
                    break
            if not stripped:
                break
        return rest, rerank, judge

    agg = {m: {"hit@1": 0, "hit@5": 0, "hit@10": 0, "mrr_sum": 0.0, "n": 0} for m in modes}
    per_item: list[dict] = []

    async with httpx.AsyncClient() as client:
        for item in items:
            row = {
                "id": item["id"],
                "question": item["question"],
                "modes": {},
            }
            for label in modes:
                mode_name, use_rerank, use_judge = _parse(label)
                data = await run_one(
                    client, base_url, item["question"], mode_name, top_k,
                    use_rerank, use_judge,
                )
                results = data.get("results", [])
                rank = first_hit_rank(results, item.get("expected_substrings", []))
                row["modes"][label] = {
                    "first_hit_rank": rank,
                    "n_results": len(results),
                    "reranked": data.get("reranked", False),
                    "llm_judged": data.get("llm_judged", False),
                    "error": data.get("error"),
                }
                a = agg[label]
                a["n"] += 1
                if rank is not None:
                    if rank <= 1:
                        a["hit@1"] += 1
                    if rank <= 5:
                        a["hit@5"] += 1
                    if rank <= 10:
                        a["hit@10"] += 1
                    a["mrr_sum"] += 1.0 / rank
            per_item.append(row)

    summary = {}
    for mode, a in agg.items():
        n = a["n"] or 1
        summary[mode] = {
            "items": a["n"],
            "hit@1": round(a["hit@1"] / n, 4),
            "hit@5": round(a["hit@5"] / n, 4),
            "hit@10": round(a["hit@10"] / n, 4),
            "mrr": round(a["mrr_sum"] / n, 4),
        }

    out = {
        "summary": summary,
        "per_item": per_item,
        "config": {"qa_path": qa_path, "base_url": base_url, "top_k": top_k, "modes": modes},
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"=== 评测汇总（n={len(items)}, top_k={top_k}）===")
    print(f"{'mode':>10}  {'Hit@1':>7}  {'Hit@5':>7}  {'Hit@10':>7}  {'MRR':>7}")
    for mode, s in summary.items():
        print(
            f"{mode:>10}  {s['hit@1']:>7.3f}  {s['hit@5']:>7.3f}  "
            f"{s['hit@10']:>7.3f}  {s['mrr']:>7.3f}"
        )
    print(f"\n详细结果写入: {out_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--qa", default="eval/qa_set.json")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument(
        "--modes",
        nargs="+",
        default=["hybrid", "vector", "keyword", "hybrid+rerank"],
        help="可用：hybrid/vector/keyword 或后缀 +rerank 强制开重排、-rerank 强制关",
    )
    p.add_argument("--out", default="eval/results/baseline.json")
    args = p.parse_args()
    return asyncio.run(
        evaluate(args.qa, args.base_url, args.top_k, args.modes, args.out)
    )


if __name__ == "__main__":
    sys.exit(main())
