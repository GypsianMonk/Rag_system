"""
Centralized configuration management using Pydantic Settings.
All values are sourced from environment variables / .env file.
"""

from functools import lru_cache
from typing import List, Literal, Optional

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────────────
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = False
    ALLOWED_ORIGINS: List[str] = ["*"]

    # ── LLM ─────────────────────────────────────────────────────────────────
    LLM_PROVIDER: Literal["openai", "anthropic", "ollama"] = "openai"
    OPENAI_API_KEY: Optional[SecretStr] = None
    ANTHROPIC_API_KEY: Optional[SecretStr] = None
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int = Field(default=2048, ge=1)
    LLM_TIMEOUT_SECONDS: int = 60

    # ── Embeddings ───────────────────────────────────────────────────────────
    EMBEDDING_PROVIDER: Literal["openai", "huggingface"] = "huggingface"
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_DIMENSION: int = 768
    EMBEDDING_BATCH_SIZE: int = 64

    # ── Vector Store ─────────────────────────────────────────────────────────
    VECTOR_STORE: Literal["faiss", "chroma", "pinecone", "weaviate"] = "faiss"
    FAISS_INDEX_PATH: str = "./data/faiss_index"
    CHROMA_PERSIST_DIR: str = "./data/chroma"
    PINECONE_API_KEY: Optional[SecretStr] = None
    PINECONE_INDEX_NAME: str = "rag-index"
    PINECONE_ENVIRONMENT: str = "us-east-1"

    # ── Retrieval ────────────────────────────────────────────────────────────
    RETRIEVAL_TOP_K: int = Field(default=5, ge=1, le=50)
    HYBRID_ALPHA: float = Field(default=0.5, ge=0.0, le=1.0)  # 0=BM25, 1=dense
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_ENABLED: bool = True
    RERANKER_TOP_K: int = Field(default=3, ge=1)
    HALLUCINATION_THRESHOLD: float = Field(default=0.5, ge=0.0, le=1.0)

    # ── Ingestion ────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = Field(default=512, ge=64)
    CHUNK_OVERLAP: int = Field(default=64, ge=0)
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: List[str] = ["pdf", "docx", "txt", "csv", "md"]

    # ── Database ─────────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://rag:rag@localhost:5432/ragdb"

    # ── Redis ────────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_SECONDS: int = 3600
    CACHE_ENABLED: bool = True

    # ── Auth / RBAC ──────────────────────────────────────────────────────────
    JWT_SECRET_KEY: SecretStr = SecretStr("change-me-in-production")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60
    AUTH_ENABLED: bool = False

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    @field_validator("CHUNK_OVERLAP")
    @classmethod
    def overlap_less_than_chunk(cls, v: int, info) -> int:
        chunk_size = info.data.get("CHUNK_SIZE", 512)
        if v >= chunk_size:
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
