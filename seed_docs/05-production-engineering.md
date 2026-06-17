# 工程化与可观测性

## 三态熔断器（Circuit Breaker）

熔断器保护下游服务（LLM API）免受连续失败冲击。三态状态机：

- **Closed（闭合）**：正常放行请求，统计连续失败次数
- **Open（打开）**：失败次数达阈值（5 次）后跳闸，**快速失败**不再调用下游，进入冷却（60 秒）
- **Half-Open（半开）**：冷却结束后允许少量试探（最多 3 次），全成功则回 Closed，任何失败回 Open

熔断器与重试是互补关系：重试解决偶发抖动，熔断防止系统性雪崩。**Open 状态下不再重试同一下游**，应改为切换备用模型或降级。

## 多模型路由

ModelRouter 维护多个模型配置（每个含 priority、weight、单独熔断器实例）。调度策略：
1. 按 priority 升序分组
2. 同 priority 内按 weight 加权随机（公式：`random() ** (1 / max(weight, 0.01))`，权重越大越容易排前）
3. 顺序尝试候选，失败则降级到下一个
4. 每个 model_id 独立熔断器，一个挂了不影响其他

加权随机用于负载均衡（避免单点过载），优先级分组用于成本/能力分级（小模型优先 + 大模型兜底）。

## SSE 流式输出

Server-Sent Events 是 HTTP 长连接，服务端用 `data: {...}\n\n` 一行一行推数据。LLM 每生成一个 token 就推一次。

效果：首字延迟（TTFT）从 5 秒降到 1 秒以内，用户感知"立即响应"；客户端可主动断开停止 token 计费。

实现：FastAPI StreamingResponse 配 `media_type="text/event-stream"`，每个 chunk 携带 trace_id，最后用 `{"done": true}` 标记结束。

## 全链路 Trace ID

每个请求入口生成一个 UUID，贯穿 Agent 编排 → 工具执行 → LLM 调用所有日志和返回。

价值：
- **故障排查**：用户报问题时附 trace_id，按 ID 一搜整条链路就出来，平均 MTTR 从小时降到分钟
- **Bad case 复盘**：用户反馈"答错了"，trace_id 能拉出当时的检索片段、工具调用、prompt
- **审计与合规**：高风险操作的完整审计链

实现风格类似 OpenTelemetry 的 span 概念，未来可直接对接 Jaeger / Langfuse 等可观测平台。

## SQLite + Chroma 持久化

文档元数据和 chunk 内容存 SQLite（异步驱动 aiosqlite）；向量存 Chroma 本地持久化目录。重启服务时自动从 SQLite 重建 BM25 内存索引，保证不丢索引。

为什么不用 PostgreSQL + Milvus：单机原型零依赖、`docker compose up` 都不用。生产场景（QPS > 50 或多副本）需要切回 PG + Milvus，但通过适配层（chroma_adapter.py 把 Chroma 伪装成 pymilvus 接口），切换时上层代码零改动。

## Token 成本控制

实测一次完整 RAG 问答约 2000 token。优化手段（按性价比）：
1. 小模型做意图分类 + 大模型只负责生成（节省最多）
2. Prompt 缓存（system prompt 固定让供应商缓存命中）
3. 检索片段精简（top-3 而非 top-10）
4. 长会话历史摘要压缩
5. 流式输出 + 用户中断（省去用户已满意时的尾部 token）
