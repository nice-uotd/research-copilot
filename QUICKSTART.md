# Research Copilot — 快速上手 & 理解指南

## 一、这个项目是干嘛的（30 秒版）

**一句话：上传文档 → 问问题 → 自动检索+生成答案。**

用户上传 PDF/Markdown → 系统切块+向量化 → 存入 Chroma 向量库
用户提问 → Agent 自动选工具（检索内部文档 / 联网搜索 / 计算器）→ 生成答案

本质上是一个 **RAG + Agent** 系统，面试时说"多工具研究助手 Agent"。

---

## 二、核心技术栈

| 组件 | 技术 | 作用 |
|------|------|------|
| Web 框架 | FastAPI | 10 个 REST API |
| 前端 | Gradio | 5-Tab Web UI |
| LLM | DeepSeek-V3（阿里百炼） | 对话生成 |
| Embedding | text-embedding-v3 | 文档向量化 |
| 向量库 | Chroma（本地） | 语义检索 |
| 关键词检索 | 自实现 BM25 | 互补语义检索 |
| 融合 | RRF（k=60）| 向量+BM25 合并排序 |
| 重排 | bge-reranker-base | Cross-Encoder 二阶段精排 |
| LLM-judge | DeepSeek 打分 | 可选的终排器（默认关） |
| Agent | OpenAI Function Calling | 自动选工具 |
| 工具 | rag_search / web_search / calculator | 三工具 |
| 数据库 | SQLite | 文档元数据 |
| 部署 | HuggingFace Spaces | 在线 demo |

---

## 三、本地跑起来（5 分钟）

### 步骤 1：安装依赖

```bash
cd /zju_0038/wyy/fujingao/research-copilot-main/research-copilot-main
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 步骤 2：配置 API Key

```bash
cp .env.example .env
```

编辑 `.env`，填入阿里百炼 key：
```
OPENAI_API_KEY=sk-你的key
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=deepseek-v3
EMBEDDING_MODEL=text-embedding-v3
```

阿里百炼注册：https://bailian.console.aliyun.com/
DeepSeek-V3 + text-embedding-v3 有免费 100 万 token 额度。

### 步骤 3：启动（两种方式）

**方式 A：完整 Web UI（推荐）**
```bash
python space_main.py
# 打开 http://127.0.0.1:7860
```

**方式 B：纯后端 API**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
# 打开 http://127.0.0.1:8000/docs 看 Swagger
```

### 步骤 4：试一试

启动后系统会自动导入 `seed_docs/` 下的 6 篇种子文档。

在 Gradio UI 上：
- Tab 1「Agent」：问 "混合检索比单路向量好在哪？" → Agent 自动调 rag_search
- Tab 2「检索对比」：输入同一查询，并排看 4 种策略结果
- Tab 3「RAG 问答」：直接 RAG 问答（不走 Agent）

### 步骤 5：跑评测（确认数字）

```bash
# 确保后端在 8000 端口运行
python eval/evaluate.py --modes hybrid+rerank hybrid-rerank vector-rerank keyword-rerank --top-k 5
```

预期输出类似：
```
hybrid+rerank:  Hit@1=0.867  MRR=0.918
hybrid-rerank:  Hit@1=0.833  MRR=0.901
vector-rerank:  Hit@1=0.800  MRR=0.874
keyword-rerank: Hit@1=0.700  MRR=0.778
```

---

## 四、关键文件地图（面试必读）

### 你必须能讲清楚的 4 个文件：

| 文件 | 干什么 | 面试怎么讲 |
|------|--------|-----------|
| `app/core/rag/retriever.py` | 向量+BM25+RRF 融合检索 | "我实现了混合检索，向量抓语义、BM25 抓关键词，RRF 融合。比单路向量 Hit@1 提升 8.4%" |
| `app/core/rag/reranker.py` | bge Cross-Encoder 重排 | "一阶段粗排后，用 Cross-Encoder 对(query,doc)对精排，净增益 +3.4pp" |
| `app/core/rag/llm_judge.py` | LLM 打分重排 | "我还试了 LLM-judge，但级联反而降 13.4pp——两者在不同子集互相否决" |
| `app/core/agent/function_calling_agent.py` | Agent 工具选择循环 | "用 OpenAI Function Calling 协议，LLM 结构化输出要调哪个工具，比文本 ReAct 更鲁棒" |

### 其他关键文件：

| 文件 | 作用 |
|------|------|
| `app/core/rag/service.py` | RAG 门面：串联 retriever → reranker → judge → generate |
| `app/config.py` | 全局配置（pydantic-settings 从 .env 读取） |
| `app/infrastructure/llm/circuit_breaker.py` | 三态熔断器 |
| `app/infrastructure/vectordb/chroma_adapter.py` | Chroma 伪装成 pymilvus 接口（适配层） |
| `app/etl/pipeline.py` | 文档解析+分块+向量化入库 |
| `space_main.py` | HF Space 入口（启动 FastAPI + 种子导入 + Gradio） |
| `eval/evaluate.py` | 评测脚本（Hit@K + MRR） |

---

## 五、核心数字（面试必背）

| 对比 | Hit@1 | MRR | 结论 |
|------|-------|-----|------|
| hybrid + rerank（默认） | 0.867 | 0.918 | 最佳配置 |
| hybrid 无重排 | 0.833 | 0.901 | rerank 净增 +3.4pp |
| 单路 vector | 0.800 | 0.874 | hybrid 比单路 +8.4% |
| keyword (BM25) | 0.700 | 0.778 | baseline |
| bge → LLM-judge 级联 | 0.733 | 0.805 | **反而降 13.4pp** |
| LLM-judge 单用 | 0.867 | 0.893 | = bge 持平，更省 |

**反直觉发现**：bge 帮了 6 题，judge 恰好搞砸这 6 题，互相否决。
**决策**：bge 默认开 + judge 改为 per-request 可选。

---

## 六、面试 STAR 故事（60 秒版）

> **S**: 做 RAG 系统时发现单路向量检索在专有名词和精确数字上效果差。
>
> **T**: 设计一个多策略检索+Agent 系统，要有量化数据证明每个组件的价值。
>
> **A**: 实现了 hybrid 检索（向量+BM25+RRF），加 bge Cross-Encoder 重排，
> 又尝试了 LLM-judge 终排。跑了 30 题×6 种配置的完整评测。
> 发现 LLM-judge 级联 bge 反而 -13.4pp——两者在不同子集互相否决。
> 最终决策：bge 默认开，judge 按请求开启。
>
> **R**: hybrid+rerank 比纯 BM25 提升 23.8%（Hit@1 0.700→0.867），
> 已部署到 HuggingFace Spaces。

---

## 七、面试追问准备

| 问题 | 要点 |
|------|------|
| 为什么不用 LangChain？ | 从零写便于理解每层原理，也方便定位 bug（比如 reranker 和 judge 冲突的问题，黑盒里很难发现） |
| RRF 的 k=60 怎么来的？ | 论文 "Reciprocal Rank Fusion outperforms Condorcet" 的经验常数，工业界标配 |
| BM25 的 k1、b 参数？ | k1=1.5 控制词频饱和度，b=0.75 控制文档长度归一化，都是经典默认值 |
| Cross-Encoder 为什么比 Bi-Encoder 准？ | Bi-Encoder 分别编码 query 和 doc 再算相似度（近似）；Cross-Encoder 把 query+doc 拼一起输入（精确交互），但慢所以只做 top-N 精排 |
| LLM-judge 为什么级联反而差？ | bge 和 judge 对"好"的定义不同：bge 偏好字面匹配，judge 偏好全面综述。某些题 bge 提上来的 judge 又压下去了 |
| Chroma 适配层怎么做的？ | 写了 `chroma_adapter.py` 实现与 pymilvus 相同的 `search()` 接口签名，上层 retriever 零改动 |
| 为什么 reranker 默认关在 HF？ | HF 免费 CPU 跑 Cross-Encoder 单次约 11s，影响体验。用户可手动开启 |

---

## 八、修复 HF Space 步骤

### 问题原因
之前的阿里百炼 API key 可能泄漏后被 revoke，或 Space 被重置。

### 修复步骤

1. **获取新的 API Key**
   - 登录 https://bailian.console.aliyun.com/
   - 创建新的 API Key（旧的如果还在，先 revoke）

2. **更新 HF Space Secrets**
   - 打开 https://huggingface.co/spaces/nice-uotd/research-copilot/settings
   - 在 "Repository secrets" 中设置：
     ```
     OPENAI_API_KEY = sk-新的key
     OPENAI_API_BASE = https://dashscope.aliyuncs.com/compatible-mode/v1
     OPENAI_MODEL = deepseek-v3
     EMBEDDING_MODEL = text-embedding-v3
     ```

3. **确保 Space 有代码**
   - 如果 Space 显示空，需要重新推送代码
   - 方法 A（GitHub Actions）：确认 GitHub repo nice-uotd/research-copilot 还在，push 一下触发 sync
   - 方法 B（手动）：
     ```bash
     # 本地克隆 HF Space（需要 HF token）
     git clone https://huggingface.co/spaces/nice-uotd/research-copilot
     # 把代码复制进去
     cp -r /zju_0038/wyy/fujingao/research-copilot-main/research-copilot-main/* research-copilot/
     cd research-copilot
     git add . && git commit -m "restore" && git push
     ```

4. **等待 Space 构建完成**（约 2-5 分钟）

5. **验证**：打开 https://huggingface.co/spaces/nice-uotd/research-copilot 看 Gradio UI 是否正常

### 注意
- HF 免费 CPU 环境下 `RERANK_ENABLED=false`（space_main.py 已自动设置）
- 首次冷启动会从 seed_docs/ 自动导入文档，约 30s
- 如果阿里百炼 key 不可用，可以换成任何 OpenAI 兼容的服务（如 SiliconFlow 的免费 DeepSeek）

---

## 九、项目不足 & 下次改进方向（面试问"还能怎么优化"时用）

1. **分块策略粗糙** — 当前按固定长度切，可以改用 SemanticChunker（按语义边界切）
2. **没有多轮记忆** — 每次问答独立，不记住上文
3. **BM25 没有持久化** — 每次重启从 SQLite 重建索引（小规模够用，大规模要 Elasticsearch）
4. **评测集自指** — 30 题问的是项目自身文档，说服力有限（可以换成真实论文集）
5. **Agent 没有流式输出** — 用户等待感知差
