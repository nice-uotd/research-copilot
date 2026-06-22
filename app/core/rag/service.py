from __future__ import annotations
import os
from pathlib import Path
from loguru import logger
from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker
from app.config import Settings
from app.core.rag.llm_judge import LLMJudge
from app.core.rag.reranker import Reranker
from app.core.rag.retriever import MultiRetriever
from app.infrastructure.database.models import DocumentChunk
from app.infrastructure.llm.embedding import EmbeddingClient
from app.infrastructure.vectordb.chroma_adapter import (
    ChromaCollectionAdapter,
    get_or_create_collection,
)
from app.models.schemas import RetrievalResult
class RAGService:
    def __init__(
        self,
        embedding: EmbeddingClient,
        vectordb: ChromaCollectionAdapter,
        retriever: MultiRetriever,
        reranker: Reranker | None = None,
        rerank_enabled: bool = True,
        llm_judge: LLMJudge | None = None,
        llm_judge_enabled: bool = False,
        oversample: int = 4,
    ) -> None:
        self.embedding = embedding
        self.vectordb = vectordb
        self.retriever = retriever
        self.reranker = reranker
        self.rerank_enabled = rerank_enabled and reranker is not None
        self.llm_judge = llm_judge
        self.llm_judge_enabled = llm_judge_enabled and llm_judge is not None
        self.oversample = max(1, oversample)
    @classmethod
    def from_settings(cls, settings: Settings, persist_dir: str = "./data/chroma") -> "RAGService":
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        if settings.hf_endpoint and not os.environ.get("HF_ENDPOINT"):
            os.environ["HF_ENDPOINT"] = settings.hf_endpoint
        embedding = EmbeddingClient(
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base or None,
            model=settings.embedding_model,
        )
        vectordb = get_or_create_collection(
            persist_dir=persist_dir,
            name=settings.milvus_collection_name,
        )
        retriever = MultiRetriever(milvus_client=vectordb, embedding_model=embedding)
        reranker: Reranker | None = None
        if settings.rerank_enabled:
            reranker = Reranker(model_name=settings.rerank_model)
            logger.info(
                "Reranker 已注册（懒加载） model={} oversample={}",
                settings.rerank_model,
                settings.retrieve_oversample,
            )
        else:
            logger.info("Reranker 已禁用 (RERANK_ENABLED=false)")
        llm_judge: LLMJudge | None = None
        if settings.openai_api_key:
            judge_client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_api_base or None,
            )
            llm_judge = LLMJudge(
                client=judge_client,
                model=settings.openai_model,
            )
            logger.info(
                "LLMJudge 已注册（per-request use_llm_judge 控制）默认 enabled={}",
                settings.llm_judge_enabled,
            )
        return cls(
            embedding=embedding,
            vectordb=vectordb,
            retriever=retriever,
            reranker=reranker,
            rerank_enabled=settings.rerank_enabled,
            llm_judge=llm_judge,
            llm_judge_enabled=settings.llm_judge_enabled,
            oversample=settings.retrieve_oversample,
        )
    async def ingest_chunks(
        self,
        chunk_ids: list[str],
        texts: list[str],
        document_id: str,
    ) -> int:
        if not chunk_ids:
            return 0
        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids 与 texts 长度必须一致")
        vectors = await self.embedding.aembed(texts)
        self.vectordb.add(
            ids=chunk_ids,
            embeddings=vectors,
            documents=texts,
            metadatas=[
                {"document_id": document_id, "chunk_index": i}
                for i in range(len(chunk_ids))
            ],
        )
        self.retriever.add_keyword_documents(dict(zip(chunk_ids, texts)))
        logger.info("RAG 入库完成 doc={} count={}", document_id, len(chunk_ids))
        return len(chunk_ids)
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        use_rerank: bool | None = None,
        use_llm_judge: bool | None = None,
    ) -> list[RetrievalResult]:
        do_bge = (
            self.rerank_enabled if use_rerank is None
            else (use_rerank and self.reranker is not None)
        )
        do_judge = (
            self.llm_judge_enabled if use_llm_judge is None
            else (use_llm_judge and self.llm_judge is not None)
        )
        if not do_bge and not do_judge:
            return await self.retriever.retrieve(query=query, top_k=top_k, mode=mode)
        candidate_k = top_k * self.oversample
        candidates = await self.retriever.retrieve(
            query=query, top_k=candidate_k, mode=mode
        )
        if not candidates:
            return []
        stage1: list[RetrievalResult] = candidates
        if do_bge:
            assert self.reranker is not None
            stage1_k = max(top_k * 2, top_k) if do_judge else top_k
            try:
                stage1 = await self.reranker.rerank(
                    query=query, documents=candidates, top_k=stage1_k
                )
                for r in stage1:
                    r.metadata = {**r.metadata, "reranked": True}
            except Exception as exc:
                logger.warning("bge 重排失败，回退原序: {}", exc)
                stage1 = candidates[:stage1_k]
        if do_judge:
            assert self.llm_judge is not None
            try:
                return await self.llm_judge.rerank(
                    query=query, documents=stage1, top_k=top_k
                )
            except Exception as exc:
                logger.warning("LLM judge 失败，回退 bge 结果: {}", exc)
                return stage1[:top_k]
        return stage1[:top_k]
    async def rebuild_bm25_from_db(
        self,
        session_factory: async_sessionmaker,
    ) -> int:
        async with session_factory() as session:
            result = await session.execute(select(DocumentChunk))
            rows = result.scalars().all()
        id_to_text = {r.id: r.content for r in rows if r.content}
        if id_to_text:
            self.retriever.register_keyword_documents(id_to_text)
        logger.info("BM25 启动重建 docs={}", len(id_to_text))
        return len(id_to_text)
_service: RAGService | None = None
def init_rag_service(settings: Settings) -> RAGService:
    global _service
    _service = RAGService.from_settings(settings)
    return _service
def get_rag_service() -> RAGService:
    if _service is None:
        raise RuntimeError("RAG 服务未初始化，请确保已在 lifespan 中调用 init_rag_service")
    return _service
