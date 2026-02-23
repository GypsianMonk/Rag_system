"""
FastAPI dependency injection — singletons wired at startup.
"""

from functools import lru_cache

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.generation.generator import RAGGenerator
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.embedder import Embedder
from app.retrieval.retriever import CrossEncoderReranker, HybridRetriever
from app.retrieval.vector_store import VectorStoreClient


@lru_cache
def _get_redis():
    if not settings.CACHE_ENABLED:
        return None
    try:
        import redis.asyncio as aioredis
        return aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    except Exception:
        return None


@lru_cache
def get_embedder() -> Embedder:
    return Embedder(cache=_get_redis())


@lru_cache
def get_vector_store() -> VectorStoreClient:
    return VectorStoreClient()


@lru_cache
def get_reranker() -> CrossEncoderReranker | None:
    if settings.RERANKER_ENABLED:
        return CrossEncoderReranker()
    return None


def get_retriever(
    embedder: Embedder = Depends(get_embedder),
    vector_store: VectorStoreClient = Depends(get_vector_store),
    reranker: CrossEncoderReranker | None = Depends(get_reranker),
) -> HybridRetriever:
    return HybridRetriever(embedder=embedder, vector_store=vector_store, reranker=reranker)


def get_generator(embedder: Embedder = Depends(get_embedder)) -> RAGGenerator:
    return RAGGenerator(embedder=embedder)


def get_ingestion_pipeline(
    embedder: Embedder = Depends(get_embedder),
    vector_store: VectorStoreClient = Depends(get_vector_store),
    db: AsyncSession = Depends(get_db),
) -> IngestionPipeline:
    return IngestionPipeline(embedder=embedder, vector_store=vector_store, db=db)


def optional_tenant(tenant_id: str | None = None) -> str | None:
    return tenant_id
