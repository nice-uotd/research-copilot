# Plan: Related Work Generator 功能

## 概要

在 Research Copilot 中新增 "Related Work Generator" 功能：用户输入研究问题/论文标题 → Agent 自动检索 arXiv + Semantic Scholar → 筛选高相关论文 → 生成一段完整的 Related Work 段落（含引用标注）。

## 设计方案

### 集成方式：新增两个 Agent 工具 + 专用 Gradio Tab

既注册为 Agent 工具（Agent 自动决定何时调用），又新增一个专用 Tab 提供针对性的 Related Work 生成体验（用定制 system prompt 引导 Agent 完成完整流程）。

### 工作流程

```
用户输入研究问题
    ↓
Agent 调用 arxiv_search(query, max_results=15)
    ↓  返回: title, authors, year, abstract, url, categories
Agent 调用 scholar_search(query, max_results=10)
    ↓  返回: title, authors, year, abstract, citation_count, url
Agent 综合两个来源，按相关性+引用量筛选 Top 8-12 篇
    ↓
Agent 生成 300-500 字的 Related Work 段落
    ↓  含 [Author, Year] 格式引用 + 末尾参考文献列表
输出完整 Related Work
```

## 新增文件

### 1. `app/core/tools/builtin/arxiv_search.py` (~130 行)

- 使用 arXiv REST API（`http://export.arxiv.org/api/query`）
- 参数: query(str), max_results(int, default=10), sort_by(str: relevance/date)
- 返回: JSON 数组，每条含 title, authors, year, abstract(前300字), arxiv_id, url, categories
- 用 httpx async 请求 + xml.etree 解析 Atom feed
- 无需 API key

### 2. `app/core/tools/builtin/scholar_search.py` (~120 行)

- 使用 Semantic Scholar Academic Graph API（`https://api.semanticscholar.org/graph/v1/paper/search`）
- 参数: query(str), max_results(int, default=10), year_min(int, optional)
- 返回: JSON 数组，每条含 title, authors, year, abstract(前300字), citation_count, url, venue
- 用 httpx async 请求
- 免费 API，无需 key（有 rate limit 100 req/5min，够用）

## 修改文件

### 3. `app/api/routes/agent.py`

- 在 `_get_agent()` 中注册 ArxivSearchTool 和 ScholarSearchTool
- 新增 `/chat-related-work` 端点，使用定制 system prompt 的 Agent 实例：
  ```
  你是一个学术写作助手。用户会给你一个研究主题，你需要：
  1. 调用 arxiv_search 和 scholar_search 搜索相关论文
  2. 从结果中筛选 8-12 篇最相关的高质量论文（优先选引用量高的）
  3. 生成一段 300-500 字的 Related Work 段落，使用 [Author, Year] 引用格式
  4. 在末尾附上参考文献列表
  ```

### 4. `gradio_app.py`

- 新增 Tab "📝 Related Work 生成"
- 输入: 研究问题/主题 (Textbox) + 论文数量上限 (Slider 5-15)
- 输出: Related Work 段落 (Markdown) + 检索到的论文列表 (Markdown) + Agent 调用轨迹
- 示例提示文字

### 5. `requirements.txt`

- 无需新增依赖（httpx 已有，xml.etree 是标准库）

## 面试亮点

1. **双源融合策略**: arXiv 覆盖最新预印本，Semantic Scholar 提供引用量信号，两者互补
2. **Agent 自主决策**: 不是硬编码 pipeline，而是让 Agent 根据中间结果决定是否需要补充搜索
3. **真实学术场景**: 不再是 toy demo，输出可直接用于论文写作
4. **免费无 key**: arXiv + Semantic Scholar 均为免费 API，HF Space 上可直接运行

## 实施顺序

1. 实现 arxiv_search 工具并单独测试
2. 实现 scholar_search 工具并单独测试
3. 修改 agent route 注册新工具 + 新增 /chat-related-work 端点
4. 新增 Gradio Tab
5. 端到端测试
