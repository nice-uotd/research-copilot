# -*- coding: utf-8 -*-
"""HuggingFace Spaces 单端口入口。

启动顺序：
  1. 设置 HF CPU 友好的环境变量默认值（Secrets 可覆盖）
  2. 创建 data/、uploads/ 目录（HF Space 临时存储）
  3. 后台 daemon 线程启动 FastAPI（127.0.0.1:8000，仅内部访问）
  4. 等 FastAPI 就绪（最多 60s）
  5. 知识库为空时自动导入 seed_docs/（冷启动）
  6. 前台启动 Gradio UI（监听 HF 期望的端口，默认 7860）

设计要点：
  - HF Spaces 只暴露一个端口（Gradio 这一侧）；FastAPI 在容器内部运行
  - gradio_app.py 内部用 BACKEND_URL=http://127.0.0.1:8000 调本地 API，零改动复用
  - HF 免费 CPU 跑 bge-reranker 单次约 11s，默认关闭重排；用户可在 UI 上按需开启
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

# --- 1. HF Space 友好的默认值（可被 Secrets 覆盖）---
os.environ.setdefault("APP_ENV", "production")
os.environ.setdefault("RERANK_ENABLED", "false")
os.environ.setdefault("LLM_JUDGE_ENABLED", "false")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./data/agent.db")
# 注意：不在此设 HF_ENDPOINT 默认值。
# - 在 HF Space 上：服务器本身就是 huggingface.co，走 hf-mirror 反而失败
# - 在本地开发：用 .env 里的 HF_ENDPOINT=https://hf-mirror.com 加速
# - 用户也可在 HF Secrets 里显式设置 HF_ENDPOINT 覆盖

# --- 2. 确保运行时目录存在 ---
Path("./data").mkdir(parents=True, exist_ok=True)
Path("./uploads").mkdir(parents=True, exist_ok=True)

import httpx
import uvicorn
from loguru import logger

# --- 3. 创建并后台启动 FastAPI ---
from app.main import create_app

fastapi_app = create_app()


def _run_api() -> None:
    """守护线程内启动 FastAPI。日志压低到 warning 减少噪声。"""
    uvicorn.run(
        fastapi_app,
        host="127.0.0.1",
        port=8000,
        log_level="warning",
    )


api_thread = threading.Thread(target=_run_api, daemon=True, name="fastapi-bg")
api_thread.start()
logger.info("FastAPI 后台线程已启动，等待就绪...")

# --- 4. 等 FastAPI 健康检查通过（最多 60s）---
_API_READY = False
for i in range(60):
    try:
        r = httpx.get("http://127.0.0.1:8000/api/v1/health", timeout=1.5)
        if r.status_code == 200:
            logger.info("FastAPI 在第 {} 秒就绪", i)
            _API_READY = True
            break
    except Exception:
        pass
    time.sleep(1)

if not _API_READY:
    logger.error("FastAPI 60s 内未就绪，Gradio 仍会启动但接口可能失败")


# --- 5. 自动导入 seed_docs/ 到知识库（冷启动）---
def _ingest_seed_docs() -> None:
    """如果知识库为空且 seed_docs/ 存在，逐个上传种子文档。"""
    seed_dir = Path("./seed_docs")
    if not seed_dir.exists():
        logger.info("没有 seed_docs/ 目录，跳过种子导入")
        return

    try:
        r = httpx.get("http://127.0.0.1:8000/api/v1/documents", timeout=10.0)
        if r.status_code == 200 and len(r.json() or []) > 0:
            logger.info("知识库已有内容（{} 文档），跳过种子导入", len(r.json()))
            return
    except Exception as e:
        logger.warning("无法检查文档列表，跳过种子导入: {}", e)
        return

    files = sorted(p for p in seed_dir.iterdir() if p.is_file())
    if not files:
        return

    logger.info("开始导入 {} 个种子文档...", len(files))
    for fp in files:
        try:
            with open(fp, "rb") as f:
                content = f.read()
            r = httpx.post(
                "http://127.0.0.1:8000/api/v1/documents/upload",
                files={"file": (fp.name, content)},
                timeout=180.0,
            )
            if r.status_code == 200:
                data = r.json()
                logger.info(
                    "种子 {} -> {} chunks", fp.name, data.get("chunk_count", "?")
                )
            else:
                logger.warning("种子 {} 上传失败 HTTP {}", fp.name, r.status_code)
        except Exception as e:
            logger.warning("种子 {} 异常: {}", fp.name, e)


if _API_READY:
    _ingest_seed_docs()


# --- 6. 启动 Gradio UI ---
os.environ.setdefault("BACKEND_URL", "http://127.0.0.1:8000")
from gradio_app import demo  # noqa: E402

if __name__ == "__main__":
    port = int(os.environ.get("PORT", os.environ.get("GRADIO_PORT", "7860")))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
    )
