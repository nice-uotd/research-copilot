# 关于 Research Copilot

## 项目概述

Research Copilot 是一个**多工具研究助手 Agent**，由 FastAPI + DeepSeek-V3 + Chroma 构建。它将三种能力组合到统一入口：基于内部知识库的 RAG 问答、联网搜索（Tavily / DDGS 双提供商）、数学计算（AST 安全求值），并由 LLM 通过 OpenAI Function Calling 协议自主路由到合适工具。

项目目标是把"能跑的 LLM 应用"做成"可观测、可容错、可上线"的工程级原型，覆盖从混合检索、二阶段重排、Agent 编排到熔断器、SSE 流式、链路追踪的完整能力栈。

## 系统架构

整体分为四层：

1. **接入层** — FastAPI 提供 11 个 HTTP 接口与 Swagger UI；Gradio 提供 5 Tab 可视化 Web UI。两者可分别对外或合并成单端口部署。
2. **领域核心层** — RAG 服务门面（service.py）装配 Embedding 客户端、向量库、多路检索器、重排器与 LLM-judge；Agent 层用 Function Calling 多步循环组合工具。
3. **基础设施层** — 多模型路由器（model_router.py）+ 三态熔断器（circuit_breaker.py）保证下游 LLM 调用韧性；SQLite + Chroma 提供本地持久化；trace_id 贯穿全链路。
4. **数据与 ETL 层** — 文档解析（pypdf / 文本读取）+ 递归切块（chunker.py）+ 向量化入库流水线。

## 核心技术栈

- **Web 框架**：FastAPI、Pydantic v2、SQLAlchemy 2.0 async、Uvicorn
- **LLM**：DeepSeek-V3（通过阿里百炼 OpenAI 兼容协议）、text-embedding-v3（1024 维）
- **向量库**：Chroma（本地持久化）+ 内存 BM25
- **重排**：BAAI/bge-reranker-base（多语言 Cross-Encoder）
- **Agent**：OpenAI Function Calling 协议 + 多步循环
- **工具**：Tavily / DDGS web search、AST 安全求值 calculator、内部 RAG 检索

## 设计取舍

项目刻意避开了几个"看起来高级但实际有坑"的方案：

1. **不用 LangChain / LangGraph**：作为学习项目从零写每个组件，便于理解每个细节和定位问题。
2. **不用文本 ReAct**：文本协议依赖 LLM 严格输出 `Thought:/Action:`，鲁棒性差。改用 OpenAI Function Calling 让 LLM 端做结构化输出。
3. **PostgreSQL → SQLite、Milvus → Chroma**：单机原型用本地存储，写适配层伪装成原接口，上层代码零改动。
4. **LLM-as-judge 默认关闭**：完整评测显示级联反而下降 13 pp，按请求开启用于推理题。

## 适合谁阅读这个项目

- 想理解"生产级 LLM 应用"长什么样的工程师
- 准备 AI Agent / LLM 应用方向面试的求职者
- 需要一个能改造的 RAG + Agent 起点骨架的开发者

完整文档与评测复现见仓库根目录的 README.md 与 eval/ 目录。
