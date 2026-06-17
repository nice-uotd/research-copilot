# 修复 HF Space + 跑评测 — 操作手册

---

## 一、本地玩（服务已跑起来）

服务在后台运行着，直接用这些命令测试：

```bash
cd /zju_0038/wyy/fujingao/research-copilot-main/research-copilot-main

# Agent 自动选工具（会消耗少量 token）
curl -s -X POST http://127.0.0.1:8000/api/v1/chat-agent \
  -H "Content-Type: application/json" \
  -d '{"query":"为什么混合检索比单路向量更好？"}'

# RAG 问答
curl -s -X POST http://127.0.0.1:8000/api/v1/chat-rag \
  -H "Content-Type: application/json" \
  -d '{"query":"Cross-Encoder和Bi-Encoder有什么区别？","top_k":3}'

# 纯检索（不生成答案）
curl -s -X POST http://127.0.0.1:8000/api/v1/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"BM25参数","top_k":5,"mode":"hybrid"}'

# Agent 联网
curl -s -X POST http://127.0.0.1:8000/api/v1/chat-agent \
  -H "Content-Type: application/json" \
  -d '{"query":"GPT-4o最新的API定价是多少？"}'

# Agent 计算
curl -s -X POST http://127.0.0.1:8000/api/v1/chat-agent \
  -H "Content-Type: application/json" \
  -d '{"query":"(1500*1 + 500*2) / 1000000 等于多少？"}'
```

如果要看 Gradio UI（需要端口转发或公网访问）：
- Gradio 端口：7860
- FastAPI Swagger：http://127.0.0.1:8000/docs

---

## 二、修复 HF Space（浏览器操作）

你的服务器连不上 huggingface.co，所以必须用浏览器+GitHub Actions。

### 步骤 1：检查 GitHub 仓库是否还在

浏览器打开：https://github.com/nice-uotd/research-copilot

- 如果能打开 → 跳到步骤 2
- 如果 404 → 需要重新创建仓库（见下面"重建仓库"部分）

### 步骤 2：更新 HF Space Secrets

浏览器打开：https://huggingface.co/spaces/nice-uotd/research-copilot/settings

在 "Repository secrets" 部分，添加/更新：

| Name | Value |
|------|-------|
| OPENAI_API_KEY | sk-33714eac6dcc4024b84db01248292298 |
| OPENAI_API_BASE | https://dashscope.aliyuncs.com/compatible-mode/v1 |
| OPENAI_MODEL | deepseek-v3 |
| EMBEDDING_MODEL | text-embedding-v3 |

### 步骤 3：确认 Space 有代码

如果 Space 显示空白/错误，需要重新推送代码。

**方法 A：通过 GitHub Actions 自动同步**

1. 打开 https://github.com/nice-uotd/research-copilot
2. 随便改一个文件（比如 README.md 加个空行）
3. Commit → 这会触发 `.github/workflows/sync-to-hf.yml`
4. 去 Actions 页面看是否成功

前提：GitHub 仓库的 Secrets 里有 `HF_TOKEN`。
检查：https://github.com/nice-uotd/research-copilot/settings/secrets/actions

如果没有 HF_TOKEN：
1. 去 https://huggingface.co/settings/tokens 创建一个新 token（权限选 Write）
2. 复制 token
3. 去 GitHub 仓库 Settings → Secrets → New repository secret
4. Name: `HF_TOKEN`，Value: 刚才的 token

**方法 B：直接推到 HF（如果 GitHub Actions 不通）**

在你有网络的电脑上（比如你的个人电脑）：
```bash
# 克隆 HF Space
git clone https://huggingface.co/spaces/nice-uotd/research-copilot
cd research-copilot

# 把代码复制进去（从 GitHub 下载的 zip 解压）
# 确保根目录有 space_main.py, gradio_app.py, app/, eval/, seed_docs/

git add .
git commit -m "restore project"
git push
```

### 步骤 4：等待构建

推送后 HF 会自动构建，约 2-5 分钟。
打开 https://huggingface.co/spaces/nice-uotd/research-copilot 查看状态。

成功后你会看到 Gradio UI 的三个 Tab。

---

## 三、跑评测

### 基础评测（不需要 reranker，快）

```bash
cd /zju_0038/wyy/fujingao/research-copilot-main/research-copilot-main

# 确保后端在跑（如果停了就重新启动）
# python space_main.py &

# 跑 3 种基础模式对比
python eval/evaluate.py --modes hybrid-rerank vector-rerank keyword-rerank --top-k 5
```

预期耗时：约 2-3 分钟（每题 1 次 API 调用做 embedding）
预期结果：
```
hybrid-rerank:  Hit@1 ≈ 0.83   MRR ≈ 0.90
vector-rerank:  Hit@1 ≈ 0.80   MRR ≈ 0.87
keyword-rerank: Hit@1 ≈ 0.70   MRR ≈ 0.78
```

### 带 reranker 评测（需要下载模型，首次慢）

先开启 reranker：
```bash
# 编辑 .env
sed -i 's/RERANK_ENABLED=false/RERANK_ENABLED=true/' .env
```

重启服务：
```bash
# 停掉旧的
pkill -f "uvicorn app.main"
sleep 2

# 重新启动
python space_main.py &
# 等约 30 秒（首次会下载 bge-reranker-base ~280MB）
sleep 30
curl -s http://127.0.0.1:8000/api/v1/health
```

跑完整评测：
```bash
python eval/evaluate.py --modes hybrid+rerank hybrid-rerank vector-rerank keyword-rerank --top-k 5
```

预期结果：
```
hybrid+rerank:  Hit@1 ≈ 0.867  MRR ≈ 0.918  ← 最佳
hybrid-rerank:  Hit@1 ≈ 0.833  MRR ≈ 0.901
vector-rerank:  Hit@1 ≈ 0.800  MRR ≈ 0.874
keyword-rerank: Hit@1 ≈ 0.700  MRR ≈ 0.778  ← baseline
```

### 带 LLM-judge 评测（慢，约 30 分钟）

```bash
python eval/evaluate.py \
  --modes "hybrid+rerank+judge" "hybrid+rerank-judge" "hybrid-rerank+judge" \
  --top-k 5
```

这会验证"级联反而降分"的反直觉发现。

---

## 四、如果重建 GitHub 仓库

如果 https://github.com/nice-uotd/research-copilot 已经不存在了：

1. 浏览器登录 GitHub (nice-uotd 账号)
2. 创建新仓库：名字 `research-copilot`，公开，不要初始化
3. 在服务器上推送（GitHub 能连）：

```bash
cd /zju_0038/wyy/fujingao/research-copilot-main/research-copilot-main
git init
git add .
git commit -m "init: Research Copilot"
git branch -M main
git remote add origin https://github.com/nice-uotd/research-copilot.git
git push -u origin main
```

4. 然后按"步骤 3 方法 A"设置 GitHub Actions 同步到 HF

---

## 五、快速验证清单

跑完以上步骤后，确认这些都 OK：

- [ ] 本地 `curl http://127.0.0.1:8000/api/v1/health` 返回 `{"status":"ok"}`
- [ ] Agent 能自动选 rag_search / web_search / calculator
- [ ] 评测结果 hybrid > vector > keyword（数字和 README 一致）
- [ ] HF Space 在线能打开（有 Gradio UI）
- [ ] GitHub 仓库公开可访问

全部 OK 后，这个项目就可以写进简历了。
