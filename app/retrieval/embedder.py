"""
Embedder — wraps OpenAI or HuggingFace embedding models with batching and caching.
"""

import asyncio
import hashlib
import json
from typing import List, Optional

import numpy as np
import structlog

from app.core.config import settings

logger = structlog.get_logger(__name__)


class Embedder:
    """Unified embedding interface with optional Redis caching."""

    def __init__(self, cache=None):
        self._cache = cache  # optional Redis client
        self._model = self._build_model()

    def _build_model(self):
        if settings.EMBEDDING_PROVIDER == "openai":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                api_key=settings.OPENAI_API_KEY.get_secret_value(),
            )
        else:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                encode_kwargs={"normalize_embeddings": True, "batch_size": settings.EMBEDDING_BATCH_SIZE},
            )

    # ── Cache helpers ────────────────────────────────────────────────────────
    def _cache_key(self, text: str) -> str:
        return f"emb:{hashlib.sha256(text.encode()).hexdigest()}"

    async def _get_cached(self, text: str) -> Optional[List[float]]:
        if not self._cache or not settings.CACHE_ENABLED:
            return None
        key = self._cache_key(text)
        val = await self._cache.get(key)
        return json.loads(val) if val else None

    async def _set_cached(self, text: str, embedding: List[float]) -> None:
        if not self._cache or not settings.CACHE_ENABLED:
            return
        key = self._cache_key(text)
        await self._cache.set(key, json.dumps(embedding), ex=settings.CACHE_TTL_SECONDS)

    # ── Public API ────────────────────────────────────────────────────────────
    async def aembed(self, text: str) -> List[float]:
        cached = await self._get_cached(text)
        if cached:
            return cached
        embedding = await asyncio.to_thread(self._model.embed_query, text)
        await self._set_cached(text, embedding)
        return embedding

    async def aembed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts, using cache where possible."""
        results: List[Optional[List[float]]] = [None] * len(texts)
        miss_indices: List[int] = []

        for i, text in enumerate(texts):
            cached = await self._get_cached(text)
            if cached:
                results[i] = cached
            else:
                miss_indices.append(i)

        if miss_indices:
            miss_texts = [texts[i] for i in miss_indices]
            # Batch embed misses
            embeddings = await asyncio.to_thread(
                self._model.embed_documents, miss_texts
            )
            for local_i, global_i in enumerate(miss_indices):
                results[global_i] = embeddings[local_i]
                await self._set_cached(texts[global_i], embeddings[local_i])

        logger.debug("batch_embedded", total=len(texts), cache_misses=len(miss_indices))
        return results  # type: ignore[return-value]

    def embed_query_sync(self, text: str) -> List[float]:
        return self._model.embed_query(text)
