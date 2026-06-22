from __future__ import annotations
import os
import threading
import time
from pathlib import Path
os.environ.setdefault("APP_ENV", "production")
os.environ.setdefault("RERANK_ENABLED", "false")
os.environ.setdefault("LLM_JUDGE_ENABLED", "false")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./data/agent.db")
Path("./data").mkdir(parents=True, exist_ok=True)
Path("./uploads").mkdir(parents=True, exist_ok=True)
import httpx
import uvicorn
from loguru import logger
from app.main import create_app
fastapi_app = create_app()
def _run_api() -> None:
    uvicorn.run(
        fastapi_app,
        host="127.0.0.1",
        port=8000,
        log_level="warning",
    )
api_thread = threading.Thread(target=_run_api, daemon=True, name="fastapi-bg")
api_thread.start()
logger.info("FastAPI 后台线程已启动，等待就绪...")
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
def _ingest_seed_docs() -> None:
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
os.environ.setdefault("BACKEND_URL", "http://127.0.0.1:8000")
from gradio_app import demo              
if __name__ == "__main__":
    port = int(os.environ.get("PORT", os.environ.get("GRADIO_PORT", "7860")))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
    )
