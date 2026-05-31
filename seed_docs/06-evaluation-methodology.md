# 评测方法论

## 为什么评测是项目的护城河

LLM 应用很容易陷入"看 demo 都对、上线都翻"的陷阱。评测脚本是把"看起来好"翻译成"数字上好"的唯一手段。本项目把评测作为一等公民：每次改 prompt、改切块策略、改重排器，都先跑评测看 Hit@K 有没有掉。

## 评测集设计

自建中英文评测集，每条 item 包含：
- `id`：唯一编号
- `lang`：zh / en
- `difficulty`：easy（字面）/ medium（同义改写、跨段、跨文档）/ hard（多跳、反例、综合）
- `question`：送入 /retrieve 接口的查询
- `expected_substrings`：命中判定依据——返回 chunk 的 content 包含其中任一子串（小写比较）即视为命中
- `expected_doc_filenames`：辅助参考，仅说明答案应来自哪些文件，不强制

难度分布建议：30% easy / 50% medium / 20% hard。简单题保证基础召回不退步，改写题考语义检索，多跳题考综合能力。

## 核心指标

- **Hit@K**：top-K 内是否命中（1 或 0），对全集求平均
- **MRR**（Mean Reciprocal Rank）：第一个命中的倒数排名（rank=1 得 1.0、rank=2 得 0.5、rank=5 得 0.2），对全集求平均
- **first_hit_rank**：每条 item 的命中名次，便于 per-item 复盘

K 的选择很关键：K 太大（如 10）容易接近饱和（所有方法都命中第 5 名内），失去区分度；K 太小（如 1）噪声大。本项目实测 K=5 在当前知识库规模下区分度最好。

## 评测命令格式

评测脚本用 `--modes` 参数指定要对比的配置，每个配置由"基础模式 + 后缀"构成：

- 基础模式：`vector` / `keyword` / `hybrid`
- 后缀（可叠加）：`+rerank` 强制开 bge / `-rerank` 强制关 bge / `+judge` 强制开 LLM-judge / `-judge` 强制关

举例：`hybrid+rerank-judge` 表示"hybrid 检索 + 强制开 bge 重排 + 强制关 LLM-judge"。

## 评测口径陷阱

第一次评测时所有模式都拿到 Hit@1=1.000——结果完全一致。原因：默认 `RERANK_ENABLED=true` 全局开关，所有模式都隐式走重排，差异化被抹掉。修复：必须用后缀强制开关，确保不同模式真的走不同路径。

这件事的教训：**评测脚本本身也要被验证**，相信"差异化数字"之前先确认实验设计本身没漏洞。

## Bad case 复盘

跑完评测保存 per_item 结果到 JSON。Bad case（first_hit_rank > 1 或 None）单独标注失效模式：
- 同源近重复段落
- 长片段稀释（关键词被均摊）
- 跨语言改写（BM25 完全失效）
- 深度推理题（cross-encoder 偏字面搞不定反例）

每类失效模式对应一种工程改进方向（父子检索、SemanticChunker、跨语言重排、LLM-as-judge）。改进上线后必须再跑回归评测确认整体不退步。

## 评测驱动决策的实例

LLM-judge 实验：烟测 1 道推理题从 rank 5 跳到 rank 1，看起来稳赢。完整 30 题评测后发现级联反而下降 13 pp（4 题受益、6 题受损）。决策：默认关闭，改为 per-request 启用。这是评测驱动决策的典型——demo 漂亮不代表真实增益。
