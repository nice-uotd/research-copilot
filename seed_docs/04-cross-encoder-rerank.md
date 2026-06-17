# Cross-Encoder 重排与 LLM-as-judge

## bi-encoder vs Cross-Encoder

向量检索和 BM25 都属于 **bi-encoder**——query 和文档**分别**编码为向量再算相似度，速度快但近似。**Cross-Encoder** 把 query 和文档拼成一对送进 BERT 类模型，输出一个标量相关性分数，慢但准。典型工业做法是二阶段：bi-encoder 召回 top-N（粗排），Cross-Encoder 精排到 top-K。

## 项目重排实现

使用 BAAI/bge-reranker-base（多语言 Cross-Encoder，约 280 MB）：

1. 检索器先用 hybrid 模式取 `top_k * oversample` 个候选（默认 oversample=4）
2. 把（query, candidate）对喂给 CrossEncoder.predict，得到分数列表
3. 按分数降序，截到 top_k 返回
4. metadata 里加上 `reranked=true` 和 `rerank_score` 便于审计

懒加载策略：模型在第一次调用时才载入内存；HF 镜像（HF_ENDPOINT=https://hf-mirror.com）解决国内拉模型慢的问题。

## CPU vs GPU 延迟

bge-reranker-base 在 CPU 上单次推理约 11 秒（候选数 20、序列长度 512），同模型在 GPU 上不到 1 秒。生产部署若 SLA 要求严格，需要 GPU 资源；演示场景可以默认关闭，按需开启。

## LLM-as-judge 终排器

在 bge 之上还可以加一层 LLM-judge：让大模型用 JSON ranking 协议给候选打分排序。优势是 LLM 能理解问题意图（不只是字面相关），对"反例"、"对比"、"原因解释"类推理题有帮助。

实现关键点：
- system prompt 强约束 JSON-only 输出
- 多层解析容错：先尝试整段 JSON，失败则用 `re.findall` 抠所有 [..] 块逐个 try
- 解析全失败时回退原始排序，**绝不阻断主链路**

## 两层重排互相否决的反直觉发现

完整评测发现：
- bge 单用相对单路向量提升 +3.4 pp Hit@1
- LLM-judge 单用与 bge 持平（且无 GPU 依赖）
- 但 **bge → judge 级联反而下降 13.4 pp**

原因：bge 偏字面相关、LLM-judge 偏意图理解，两者擅长的子集不同。bge 已经做对的字面题，LLM-judge 反而把"全面综述段"换成"字面命中段"，损失精确性。

最终设计：bge 默认开启；LLM-judge 改为按请求开启（use_llm_judge=true），用于推理类 bad case，不污染字面题的多数场景。这个决策的关键是**评测先行，不靠看着 demo 漂亮就上线**。
