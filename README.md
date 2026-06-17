---
title: Research Copilot
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: space_main.py
pinned: false
license: mit
short_description: 多工具研究助手 Agent · 混合检索 + 重排 + Function Calling
---

# Research Copilot — 多工具研究助手 Agent

> 基于 **FastAPI + DeepSeek-V3 + Chroma** 的工程级 LLM 应用。
> **混合检索（向量 + BM25 + RRF）+ Cross-Encoder 二阶段重排 + Function Calling Agent 三工具自主路由**，
> 配套 30 条评测集量化对比 4 种检索策略。

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![FastAPI](https://img.shields.io/badge/fastapi-0.115%2B-009688)]()
[![Gradio](https://img.shields.io/badge/gradio-6.x-orange)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## 在线 Demo

如果你正在 HuggingFace Space 上看到这个 README，UI 已经在上方运行——直接试三个 Tab：

- **🤖 Agent**：自主路由 RAG / 联网 / 计算
- **⚖️ 检索对比**：同一查询并排跑 4 种策略（最直观看出混合检索价值）
- **📚 RAG 问答**：检索内部文档并带引用作答

---

## 核心数字（30 条评测集，top_k=5）

| 模式 | Hit@1 | Hit@5 | MRR | 备注 |
|---|---:|---:|---:|---|
| **hybrid + rerank** | **0.867** | 1.000 | **0.918** | 默认部署配置 |
| hybrid（无重排）| 0.833 | 1.000 | 0.901 | 重排净增益 +3.4 pp |
| vector（无重排）| 0.800 | 1.000 | 0.874 | hybrid 比单路 +8.4% |
| keyword (BM25) | 0.700 | 0.867 | 0.778 | hybrid+rerank 相对此 **+23.8%** |

**反直觉对比实验**（数据驱动取舍而非炫技）：
- LLM-as-judge 单独使用 = bge 持平（0.867），且更省（无 GPU、无 280MB 模型）
- 但 **bge → judge 级联反而下降至 0.733**（-13.4 pp，两者在不同子集互相否决）
- 最终决策：bge 默认开启 + LLM-judge 改为按请求开启（`use_llm_judge=true`）

---

## 系统架构

```
              ┌──────────────────────────────────────────┐
              │  FastAPI / Swagger UI · Gradio 5-Tab UI  │
              └────────────────┬─────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────────┐
        ▼                      ▼                          ▼
 /chat /chat/stream      /chat-rag /retrieve         /chat-agent
 (SSE 流式 + 熔断)       (混合检索 + 重排 + 引用)    (Function Calling)
        │                      │                          │
        │                      │           ┌──────────────┼──────────┐
        │                      │           ▼              ▼          ▼
        │                      │      rag_search     web_search   calculator
        │                      │                     (Tavily +    (AST 安全
        │                      │                      DDGS)        求值)
        │                      ▼
        │              ┌────────────────┐
        │              │ MultiRetriever │
        │              │  ├ Vector      │ → text-embedding-v3 → Chroma
        │              │  ├ BM25        │
        │              │  └ RRF Fusion  │
        │              └────┬───────────┘
        │                   │
        │                   ▼
        │       bge-reranker-base (Cross-Encoder)
        │           ↓ (per-request)
        │       LLM-as-judge
        ▼
 ModelRouter (三态熔断器)
        │
        ▼
 DeepSeek-V3 (阿里百炼 OpenAI 兼容)
```

**持久化**：SQLite（文档元数据 + chunk）+ Chroma（向量库），重启自动恢复 BM25 索引。

---

## 5 分钟跑通

### 前置

- Python 3.10+
- 阿里百炼 API Key（[bailian.console.aliyun.com](https://bailian.console.aliyun.com/)，DeepSeek-V3 + text-embedding-v3 免费 100 万 token）

### 安装

```bash
git clone https://github.com/<your-username>/research-copilot.git
cd research-copilot

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# 编辑 .env：填入 OPENAI_API_KEY=sk-xxx
```

### 启动方式 A：纯后端（FastAPI + Swagger）

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

打开 <http://127.0.0.1:8000/> 自动跳到 Swagger UI。

### 启动方式 B：完整 Web UI（FastAPI + Gradio 同进程）

```bash
python space_main.py
```

打开 <http://127.0.0.1:7860/> 看 5 Tab Web UI。

### 试一下

```bash
# 1. 健康检查
curl http://127.0.0.1:8000/api/v1/health

# 2. 上传文档（自动切块 + 向量化）
curl -X POST http://127.0.0.1:8000/api/v1/documents/upload \
  -F "file=@your_paper.pdf"

# 3. 混合检索 + 重排
curl -X POST http://127.0.0.1:8000/api/v1/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"你的问题","top_k":3,"mode":"hybrid","use_rerank":true}'

# 4. RAG 问答（带引用）
curl -X POST http://127.0.0.1:8000/api/v1/chat-rag \
  -H "Content-Type: application/json" \
  -d '{"query":"你的问题","top_k":3}'

# 5. Agent 自主路由（RAG / 联网 / 计算 三工具）
curl -X POST http://127.0.0.1:8000/api/v1/chat-agent \
  -H "Content-Type: application/json" \
  -d '{"query":"DeepSeek-V3 输入 1500 + 输出 500 token 按 GPT-4o 价格算多少美元？"}'
```

---

## API 一览

| 接口 | 方法 | 说明 |
|---|---|---|
| `/api/v1/health` | GET | 存活探针 |
| `/api/v1/chat` | POST | 裸 LLM 对话（多模型路由 + 熔断）|
| `/api/v1/chat/stream` | POST | SSE 流式对话 |
| `/api/v1/documents/upload` | POST | 文档上传 + 自动嵌入 |
| `/api/v1/documents` | GET | 文档列表 |
| `/api/v1/retrieve` | POST | 检索（vector/keyword/hybrid + use_rerank/use_llm_judge）|
| `/api/v1/chat-rag` | POST | RAG 问答（检索 → 注入 → 生成 + 引用解析）|
| `/api/v1/web-search` | POST | 联网搜索（Tavily 优先 + DDGS 回退）|
| `/api/v1/chat-web` | POST | 联网 RAG（搜索 + 注入 + 生成）|
| `/api/v1/chat-agent` | POST | Function Calling Agent 三工具自主路由 |

每个接口都返回 `trace_id`，全链路可追踪。

---

## 评测复现

```bash
# 4 模式基线（约 9 分钟）
python eval/evaluate.py \
  --modes hybrid-rerank hybrid+rerank vector-rerank keyword-rerank \
  --top-k 5

# 含 LLM-judge 三模式（约 28 分钟，会调 30+ 次 LLM）
python eval/evaluate.py \
  --modes "hybrid+rerank-judge" "hybrid+rerank+judge" "hybrid-rerank+judge" \
  --top-k 5
```

后缀语义：`+rerank/-rerank` 强制开/关 bge 重排；`+judge/-judge` 强制开/关 LLM-judge。
不带后缀跟随 `.env` 全局开关。

---

## 项目结构

```
research-copilot/
├── space_main.py             # HF Space 入口（FastAPI + Gradio 同进程）
├── gradio_app.py             # 5 Tab Web UI（HTTP 调本地 FastAPI）
├── app/
│   ├── main.py               # FastAPI 入口、lifespan、根路径跳转
│   ├── config.py             # pydantic-settings 配置
│   ├── api/routes/           # health / chat / document / rag / web / agent
│   ├── core/
│   │   ├── rag/
│   │   │   ├── service.py    # 三段流水线门面
│   │   │   ├── retriever.py  # 向量+BM25+RRF
│   │   │   ├── reranker.py   # bge-reranker-base 包装
│   │   │   └── llm_judge.py  # LLM-as-judge JSON ranking
│   │   ├── agent/
│   │   │   └── function_calling_agent.py  # OpenAI tools 协议循环
│   │   └── tools/
│   │       └── builtin/{search,calculator}.py
│   ├── infrastructure/
│   │   ├── llm/{model_router,circuit_breaker,embedding}.py
│   │   ├── vectordb/chroma_adapter.py     # Chroma 伪装 pymilvus 接口
│   │   ├── database/{models,session}.py   # SQLAlchemy ORM
│   │   └── trace/tracer.py                # 链路追踪
│   └── etl/{parser,chunker,pipeline}.py   # 解析 + 分块 + 流水线
├── eval/
│   ├── qa_set.json           # 公开评测集（自指型 demo）
│   └── evaluate.py           # 评测脚本（Hit@K + MRR）
├── seed_docs/                # 项目自指型种子文档（启动时自动入库）
└── data/                     # SQLite + Chroma 持久化（gitignore）
```

---

## 技术栈

```
Python 3.10 · FastAPI · Pydantic v2 · SQLAlchemy 2.0 (async) · Gradio
LLM: DeepSeek-V3 (阿里百炼 OpenAI 兼容) · text-embedding-v3 (1024 维)
向量库: Chroma (本地持久化) · BM25 (内存)
重排: BAAI/bge-reranker-base (Cross-Encoder, 多语言)
Agent: OpenAI Function Calling 协议 · 多步循环
工具: Tavily + DDGS (web search) · AST safe-eval (calculator)
工程: SSE 流式 · 三态熔断 · trace_id 全链路 · loguru
```

---

## HuggingFace Space 部署说明

`space_main.py` 是单端口入口：

1. 后台守护线程启动 FastAPI（`127.0.0.1:8000`，仅容器内访问）
2. 等就绪后自动从 `seed_docs/` 导入种子文档（HF 临时存储重启会丢，每次冷启动重建）
3. 前台启动 Gradio UI（HF 期望端口）

HF Space Secrets（在 Space Settings 页面配置）：

- `OPENAI_API_KEY`（必填）
- `OPENAI_API_BASE`、`OPENAI_MODEL`、`EMBEDDING_MODEL`（可选，有默认值）

HF 免费 CPU 跑 bge-reranker 单次约 11s，故 `RERANK_ENABLED=false` 默认关闭，用户可在 UI 上按需开启。

---

## 设计取舍（面试可深挖）

1. **不用 LangChain / LangGraph**：从零写每个组件，便于理解原理与定位问题
2. **不用文本 ReAct**：改用 OpenAI Function Calling 让 LLM 端做结构化输出，鲁棒性大幅提升
3. **PostgreSQL → SQLite、Milvus → Chroma**：单机原型用本地存储；写适配层伪装成原接口，上层零改动
4. **LLM-as-judge 默认关闭**：完整评测显示级联反而 -13.4 pp，按请求开启用于推理题——数据驱动取舍而非炫技

---

## 后续路线

- [x] Cross-Encoder 二阶段重排
- [x] LLM-as-judge 三段流水线（按请求开启）
- [x] Gradio Web UI
- [x] HuggingFace Spaces 部署
- [ ] 父子检索 / SemanticChunker（修复近重复段落 Bad case）
- [ ] Agent SSE 流式（步骤实时返回）
- [ ] 多轮记忆（短期 Redis + 长期向量库摘要）

---

## License

MIT
