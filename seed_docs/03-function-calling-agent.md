# Function Calling Agent 设计

## 为什么不用文本式 ReAct

文本式 ReAct（Thought / Action / Action Input / Observation 循环）依赖 LLM 严格按格式输出，由应用侧用正则解析。常见问题：

- LLM 偶尔输出"思考: ..."（中文冒号）或"Thinking:"（不同语言风格）就解析失败
- 不同模型的输出风格不一致，需要为每个模型维护一套解析规则
- 每轮 prompt 携带完整 ReAct 模板，token 开销大

OpenAI Function Calling 协议在 2023 年成为工业标准，DeepSeek、通义千问、Claude、GPT 全部原生支持。优势：LLM 端做结构化输出（机器可解析的 tool_calls），应用层只需 JSON 解析，鲁棒性大幅提升。

## 协议工作流程

1. **首轮**：客户端把工具定义（name + description + parameters JSON Schema）和用户消息一起发给模型
2. **决策**：模型不直接答，而是返回 `tool_calls = [{id, function: {name, arguments}}, ...]`
3. **执行**：应用按 name 路由到本地函数，arguments 反序列化后真正调用
4. **回灌**：把每个工具结果作为 `role="tool"` 消息（带 tool_call_id）追加到对话
5. **次轮**：再调模型，模型综合工具结果生成最终答案
6. **终止条件**：模型不再返回 tool_calls（说明它要直接答）或达到 max_iters 上限

## 项目工具清单

- **rag_search**：搜索内部向量库（混合检索 + 可选重排），返回 JSON 数组（id、score、content）
- **web_search**：联网搜索，Tavily 优先 + DDGS 回退；返回 JSON（title、url、content、score、source）
- **calculator**：数学表达式求值，**关键安全设计**——用 ast.parse 解析后白名单只允许 BinOp / UnaryOp / Constant，禁止任意函数调用，防止 LLM 通过工具触发 RCE

## 工具描述如何写

description 是 LLM 决策的主要依据。好的写法包含：
- 写明工具领域和典型场景
- 给术语锚点（像"询问 X、Y、Z 类问题时使用"）
- 隐含边界（言外之意：不在这些场景就别用）

示例：rag_search 的描述里同时提到了"项目文档、技术文档、Agent/RAG 设计资料"作为正面引导，让 LLM 在这些主题下倾向选它。

## 多步循环与防死循环

Agent 循环最多 max_iters 步（默认 5）。每步执行后 observation 被截断到 1500 字符以避免上下文炸裂；完整结果（最长 6000 字符）回灌到对话以保证模型能看到关键信息。

死循环防护：硬上限 max_iters；后续可加重复检测（连续两次相同 tool_name+args_hash 就提前中断）和观察增长检测。

## 与 RAG 的关系

Function Calling 是"模型选择动作"的上层协议，RAG 是"检索再生成"的具体动作。在本项目里 RAG 被包装成 rag_search 工具，由 Agent 决定何时调用。两者是组合关系不是互斥关系。
