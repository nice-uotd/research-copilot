"""应用配置：基于 pydantic-settings，支持环境变量与 .env 文件。"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="enterprise-ai-agent", description="服务名称")
    app_env: str = Field(default="development", description="运行环境")
    debug: bool = Field(default=False, description="调试模式")
    api_prefix: str = Field(default="/api/v1", description="API 前缀")
    host: str = Field(default="0.0.0.0", description="监听地址")
    port: int = Field(default=8000, description="监听端口")

    openai_api_key: str = Field(default="", description="OpenAI API Key")
    openai_api_base: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI 兼容 API Base",
    )
    openai_model: str = Field(default="gpt-4o-mini", description="默认对话模型")

    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/agent_db",
        description="SQLAlchemy 异步数据库 URL（推荐 postgresql+asyncpg）",
    )
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis URL")

    milvus_host: str = Field(default="localhost", description="Milvus 主机")
    milvus_port: int = Field(default=19530, description="Milvus 端口")
    milvus_user: str = Field(default="", description="Milvus 用户名")
    milvus_password: str = Field(default="", description="Milvus 密码")
    milvus_collection_name: str = Field(
        default="agent_knowledge",
        description="默认向量集合名",
    )

    embedding_model: str = Field(
        default="text-embedding-v3",
        description="Embedding 模型 ID（OpenAI 兼容协议）",
    )

    rerank_model: str = Field(
        default="BAAI/bge-reranker-base",
        description="Cross-Encoder 重排模型（HuggingFace ID 或本地路径）",
    )
    rerank_enabled: bool = Field(
        default=True,
        description="重排器全局开关；关闭后 use_rerank=true 也会被忽略",
    )
    retrieve_oversample: int = Field(
        default=4,
        ge=1,
        description="重排开启时，初检候选数 = top_k * oversample",
    )

    llm_judge_enabled: bool = Field(
        default=False,
        description="LLM-as-judge 全局默认开关；可在 /retrieve 上用 use_llm_judge 覆盖",
    )
    hf_endpoint: str = Field(
        default="https://hf-mirror.com",
        description="HuggingFace 镜像端点（国内拉模型用）",
    )

    tavily_api_key: str = Field(
        default="",
        description="Tavily 搜索 API Key；为空则自动回退到 DDGS",
    )
    web_search_max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="单次 web search 返回结果上限",
    )
    web_search_timeout: float = Field(
        default=15.0,
        ge=1.0,
        description="web search 超时时间（秒）",
    )

    log_level: str = Field(default="INFO", description="日志级别")


@lru_cache
def get_settings() -> Settings:
    return Settings()
