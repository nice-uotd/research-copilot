from __future__ import annotations
import json
import os
from typing import Any
import gradio as gr
import httpx
BACKEND = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
API = f"{BACKEND}/api/v1"
def _post(path: str, payload: dict, timeout: float = 180.0) -> dict:
    try:
        r = httpx.post(f"{API}{path}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        body = e.response.text[:300] if e.response else ""
        return {"_error": f"HTTP {e.response.status_code}: {body}"}
    except httpx.ConnectError:
        return {"_error": f"连接 {BACKEND} 失败，请确认 FastAPI 服务已启动"}
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}"}
def agent_chat(query: str, max_iters: int) -> tuple[str, str, str]:
    if not query or not query.strip():
        return "请输入问题", "", ""
    data = _post("/chat-agent", {"query": query, "max_iters": int(max_iters)})
    if "_error" in data:
        return f"❌ {data['_error']}", "", ""
    answer = data.get("answer", "(空)")
    steps = data.get("steps", [])
    iters = data.get("iterations", 0)
    tokens = data.get("total_tokens", 0)
    finished = data.get("finished", False)
    trace_id = data.get("trace_id", "")
    meta = (
        f"**状态**: {'✅ 完成' if finished else '⚠️ 未完成'}  ·  "
        f"**迭代**: {iters}  ·  **Token**: {tokens}  ·  "
        f"**Trace**: `{trace_id[:8]}...`"
    )
    if not steps:
        trace_md = "_本次未调用任何工具，模型直接作答_"
    else:
        lines = [f"### 工具调用轨迹（共 {len(steps)} 步）\n"]
        for s in steps:
            args_str = json.dumps(s.get("tool_args", {}), ensure_ascii=False)
            lines.append(f"**Step {s['step']}** — `{s['tool_name']}`")
            lines.append(f"- 参数: `{args_str}`")
            result = s.get("tool_result", "")
            preview = result[:400] + ("..." if len(result) > 400 else "")
            lines.append(f"- 返回:\n  ```\n  {preview}\n  ```")
            if s.get("error"):
                lines.append(f"- ⚠️ 错误: `{s['error']}`")
            lines.append("")
        trace_md = "\n".join(lines)
    return answer, trace_md, meta
def rag_qa(query: str, top_k: int, mode: str, rerank_choice: str) -> tuple[str, str]:
    if not query or not query.strip():
        return "请输入问题", ""
    payload: dict[str, Any] = {"query": query, "top_k": int(top_k), "mode": mode}
    if rerank_choice == "强制开启":
        payload["use_rerank"] = True
    elif rerank_choice == "强制关闭":
        payload["use_rerank"] = False
    data = _post("/chat-rag", payload)
    if "_error" in data:
        return f"❌ {data['_error']}", ""
    answer = data.get("answer", "(空)")
    contexts = data.get("contexts", [])
    usage = data.get("usage") or {}
    lines = [
        f"### 检索 {len(contexts)} 段  ·  model={data.get('model','?')}  "
        f"·  tokens={usage.get('total_tokens','?')}  ·  trace=`{data.get('trace_id','')[:8]}...`\n"
    ]
    for i, c in enumerate(contexts, 1):
        score = c.get("score", 0.0)
        meta = c.get("metadata") or {}
        rerank_tag = " 🎯reranked" if meta.get("reranked") else ""
        src = c.get("source", "?")
        content = c.get("content", "")
        preview = content[:400] + ("..." if len(content) > 400 else "")
        lines.append(
            f"**[{i}]** id=`{c.get('id','')[:8]}` score={score:.4f} src=`{src}`{rerank_tag}"
        )
        lines.append(f"```\n{preview}\n```")
    return answer, "\n".join(lines)
def _format_retrieval(label: str, data: dict) -> str:
    if "_error" in data:
        return f"### {label}\n\n❌ {data['_error']}"
    items = data.get("results", [])
    if not items:
        return f"### {label}\n\n_无结果_"
    out = [f"### {label}  ({len(items)} 条)\n"]
    for i, it in enumerate(items, 1):
        score = it.get("score", 0.0)
        content = it.get("content", "").replace("\n", " ")
        preview = content[:200] + ("..." if len(content) > 200 else "")
        out.append(f"**[{i}]** id=`{it.get('id','')[:8]}` score={score:.4f}")
        out.append(f"> {preview}\n")
    return "\n".join(out)
def retrieval_compare(query: str, top_k: int) -> tuple[str, str, str, str]:
    if not query or not query.strip():
        empty = "请输入问题"
        return empty, empty, empty, empty
    configs = [
        ("hybrid + rerank", {"mode": "hybrid", "use_rerank": True}),
        ("hybrid (无重排)", {"mode": "hybrid", "use_rerank": False}),
        ("vector 单路", {"mode": "vector", "use_rerank": False}),
        ("keyword (BM25)", {"mode": "keyword", "use_rerank": False}),
    ]
    out: list[str] = []
    for label, opts in configs:
        payload = {"query": query, "top_k": int(top_k), **opts}
        data = _post("/retrieve", payload)
        out.append(_format_retrieval(label, data))
    return out[0], out[1], out[2], out[3]
def web_qa(query: str, max_results: int) -> tuple[str, str]:
    if not query or not query.strip():
        return "请输入问题", ""
    data = _post(
        "/chat-web",
        {"query": query, "max_results": int(max_results)},
        timeout=120.0,
    )
    if "_error" in data:
        return f"❌ {data['_error']}", ""
    answer = data.get("answer", "(空)")
    contexts = data.get("contexts", [])
    provider = data.get("provider", "?")
    usage = data.get("usage") or {}
    lines = [
        f"### {len(contexts)} 个网页  ·  provider=`{provider}`  ·  "
        f"tokens={usage.get('total_tokens','?')}\n"
    ]
    for i, c in enumerate(contexts, 1):
        title = c.get("title", "")[:80]
        url = c.get("url", "")
        content = c.get("content", "")[:250]
        score = c.get("score", 0.0)
        lines.append(f"**[{i}] {title}**  · score={score:.3f}")
        lines.append(f"🔗 {url}")
        lines.append(f"> {content}...\n")
    return answer, "\n".join(lines)
INTRO = """
# 🤖 Research Copilot — 多工具研究助手 Agent
**五工具自主路由**（知识库 RAG · 联网搜索 · 数学计算 · arXiv 论文检索 · Semantic Scholar）  ·
**混合检索 + 重排**（向量 + BM25 + RRF + bge-reranker）  ·
**Related Work 自动生成**（双源检索 → 筛选 → 综述段落）  ·
**真实评测**：30 条评测集，hybrid+rerank Hit@1 = **0.867**，比 BM25 **+23.8%**
> 💡 **使用说明**：上方 5 个 Tab 演示核心能力；文档上传请用 [Swagger API 文档](/docs)（HF 已自动加载 6 篇种子文档）。
"""
EXAMPLES_AGENT_TXT = """
**🎯 试试这些问题**（复制下方一行到上面文本框）：
- `RRF 倒数排名融合的常数 k 取多少？为什么是这个值？` ← 走 rag_search
- `2025 年 LangGraph 最新版本号是多少？` ← 走 web_search
- `输入 1500 + 输出 500 token 按 input $5/M output $15/M 算多少美元？` ← 走 calculator
- `Search arXiv for recent papers on retrieval augmented generation` ← 走 arxiv_search
- `Find highly cited papers on multi-agent systems` ← 走 scholar_search
"""
EXAMPLES_RAG_TXT = """
**🎯 试试这些问题**：
- `为什么混合检索比单路向量更稳？`
- `三态熔断器的恢复窗口设置成多少秒？`
- `为什么 LLM-as-judge 加在 bge 之后反而下降？`
"""
EXAMPLES_CMP_TXT = """
**🎯 短查询效果最直观**：`RRF 排名融合` / `Cross-Encoder 重排` / `熔断器三态`
"""
EXAMPLES_WEB_TXT = """
**🎯 试试**：`2025 LangGraph 最新版本` / `OpenAI 最新模型定价`
"""
EXAMPLES_RW_TXT = """
**🎯 试试这些研究主题**（英文效果最佳）：
- `Retrieval-Augmented Generation for knowledge-intensive tasks`
- `Multi-agent collaboration and communication in LLM systems`
- `Active learning for knowledge graph alignment`
- `Circuit breaker patterns in LLM serving systems`
"""
def related_work_gen(topic: str, max_papers: int) -> tuple[str, str, str]:
    if not topic or not topic.strip():
        return "请输入研究主题", "", ""
    data = _post(
        "/chat-related-work",
        {"topic": topic, "max_papers": int(max_papers)},
        timeout=300.0,                                
    )
    if "_error" in data:
        return f"❌ {data['_error']}", "", ""
    answer = data.get("related_work", "(空)")
    steps = data.get("steps", [])
    iters = data.get("iterations", 0)
    tokens = data.get("total_tokens", 0)
    finished = data.get("finished", False)
    trace_id = data.get("trace_id", "")
    papers_found = data.get("papers_found", 0)
    meta = (
        f"**状态**: {'✅ 完成' if finished else '⚠️ 未完成'}  ·  "
        f"**迭代**: {iters}  ·  **Token**: {tokens}  ·  "
        f"**检索论文数**: {papers_found}  ·  "
        f"**Trace**: `{trace_id[:8]}...`"
    )
    if not steps:
        trace_md = "_未调用工具_"
    else:
        lines = [f"### 检索轨迹（共 {len(steps)} 步）\n"]
        for s in steps:
            args_str = json.dumps(s.get("tool_args", {}), ensure_ascii=False)
            lines.append(f"**Step {s['step']}** — `{s['tool_name']}`")
            lines.append(f"- 参数: `{args_str}`")
            result = s.get("tool_result", "")
            if s["tool_name"] in ("arxiv_search", "scholar_search"):
                try:
                    papers = json.loads(result)
                    if isinstance(papers, list):
                        titles = [
                            f"  - {p.get('title', '?')} ({p.get('year', '?')})"
                            + (f" [cited: {p['citation_count']}]" if 'citation_count' in p else "")
                            for p in papers[:8]
                        ]
                        preview = "\n".join(titles)
                        if len(papers) > 8:
                            preview += f"\n  - ... 共 {len(papers)} 篇"
                        lines.append(f"- 论文列表:\n{preview}")
                    else:
                        lines.append(f"- 返回: `{result[:300]}...`")
                except (json.JSONDecodeError, TypeError):
                    lines.append(f"- 返回: `{result[:300]}...`")
            else:
                preview = result[:300] + ("..." if len(result) > 300 else "")
                lines.append(f"- 返回: `{preview}`")
            if s.get("error"):
                lines.append(f"- ⚠️ 错误: `{s['error']}`")
            lines.append("")
        trace_md = "\n".join(lines)
    return answer, trace_md, meta
with gr.Blocks(title="Research Copilot", theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO)
    with gr.Tab("🤖 Agent（自主路由）"):
        gr.Markdown(
            "Agent 自主选择工具：**知识库题用 RAG · 时效题用搜索 · 数学题用计算器**。"
        )
        gr.Markdown(EXAMPLES_AGENT_TXT)
        with gr.Row():
            with gr.Column(scale=3):
                agent_q = gr.Textbox(label="问题", lines=2, placeholder="问任何问题…")
            with gr.Column(scale=1):
                agent_iters = gr.Slider(1, 8, value=4, step=1, label="最大循环次数")
        agent_btn = gr.Button("提交", variant="primary")
        agent_meta = gr.Markdown()
        agent_answer = gr.Markdown()
        gr.Markdown("### 🔍 工具调用轨迹")
        agent_trace = gr.Markdown()
        agent_btn.click(
            agent_chat,
            [agent_q, agent_iters],
            [agent_answer, agent_trace, agent_meta],
        )
    with gr.Tab("📚 RAG 问答（带引用）"):
        gr.Markdown("检索内部知识库 → 注入上下文 → LLM 生成 → 解析 [n] 引用")
        gr.Markdown(EXAMPLES_RAG_TXT)
        with gr.Row():
            with gr.Column(scale=3):
                rag_q = gr.Textbox(label="问题", lines=2)
            with gr.Column(scale=1):
                rag_top_k = gr.Slider(1, 8, value=3, step=1, label="top_k")
                rag_mode = gr.Dropdown(
                    ["hybrid", "vector", "keyword"], value="hybrid", label="检索模式"
                )
                rag_rerank = gr.Dropdown(
                    ["默认（跟全局）", "强制开启", "强制关闭"],
                    value="默认（跟全局）",
                    label="Cross-Encoder 重排",
                )
        rag_btn = gr.Button("提交", variant="primary")
        rag_answer = gr.Markdown()
        gr.Markdown("### 📖 检索片段")
        rag_ctx = gr.Markdown()
        rag_btn.click(
            rag_qa,
            [rag_q, rag_top_k, rag_mode, rag_rerank],
            [rag_answer, rag_ctx],
        )
    with gr.Tab("⚖️ 检索对比（4 模式并排）"):
        gr.Markdown(
            "**项目核心卖点**：同一查询用 4 种策略并排跑，"
            "直观看出 hybrid+rerank 比单路 BM25 / vector 强多少。"
        )
        gr.Markdown(EXAMPLES_CMP_TXT)
        with gr.Row():
            cmp_q = gr.Textbox(label="问题", lines=2, scale=3)
            cmp_top_k = gr.Slider(1, 6, value=3, step=1, label="top_k", scale=1)
        cmp_btn = gr.Button("并排运行", variant="primary")
        with gr.Row():
            cmp_a = gr.Markdown()
            cmp_b = gr.Markdown()
        with gr.Row():
            cmp_c = gr.Markdown()
            cmp_d = gr.Markdown()
        cmp_btn.click(
            retrieval_compare, [cmp_q, cmp_top_k], [cmp_a, cmp_b, cmp_c, cmp_d]
        )
    with gr.Tab("🌐 联网搜索 + RAG"):
        gr.Markdown("Tavily（如配 key）/ DDGS 搜索 → LLM 基于摘要作答")
        gr.Markdown(EXAMPLES_WEB_TXT)
        with gr.Row():
            web_q = gr.Textbox(label="问题", lines=2, scale=3)
            web_n = gr.Slider(1, 6, value=3, step=1, label="结果数", scale=1)
        web_btn = gr.Button("联网搜索 + RAG", variant="primary")
        web_answer = gr.Markdown()
        gr.Markdown("### 🔗 网页摘要")
        web_ctx = gr.Markdown()
        web_btn.click(web_qa, [web_q, web_n], [web_answer, web_ctx])
    with gr.Tab("📝 Related Work 生成"):
        gr.Markdown(
            "**核心功能**：输入研究主题 → Agent 自动检索 arXiv + Semantic Scholar "
            "→ 筛选高相关论文 → 生成完整 Related Work 段落（含引用标注）。\n\n"
            "适用场景：论文写作初期快速了解相关工作、生成 Related Work 草稿。"
        )
        gr.Markdown(EXAMPLES_RW_TXT)
        with gr.Row():
            with gr.Column(scale=3):
                rw_topic = gr.Textbox(
                    label="研究主题",
                    lines=2,
                    placeholder="输入研究问题或论文标题（英文效果更佳）…",
                )
            with gr.Column(scale=1):
                rw_papers = gr.Slider(
                    5, 15, value=10, step=1, label="参考论文数量"
                )
        rw_btn = gr.Button("生成 Related Work", variant="primary")
        rw_meta = gr.Markdown()
        gr.Markdown("### 📄 Generated Related Work")
        rw_answer = gr.Markdown()
        gr.Markdown("### 🔍 Agent 检索轨迹")
        rw_trace = gr.Markdown()
        rw_btn.click(
            related_work_gen,
            [rw_topic, rw_papers],
            [rw_answer, rw_trace, rw_meta],
        )
    gr.Markdown(
        f"---\n后端: `{BACKEND}`  ·  "
        f"[Swagger API 文档]({BACKEND}/docs)  ·  "
        f"开源仓库: [GitHub](https://github.com/nice-uotd/research-copilot)"
    )
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_PORT", "7860")),
        share=False,
        show_error=True,
    )
