"""
Shared test fixtures and mocks.

Patches heavy dependencies (LLM, embedder, DB, vector store) so all tests
run in CI without real API keys, model downloads, or a live database.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.core.database import get_db
from app.main import app
from app.retrieval.vector_store import SearchResult
from app.utils.dependencies import get_embedder, get_generator, get_reranker, get_vector_store

_DIM = 768
_FAKE_EMB: list[float] = (np.ones(_DIM) / np.sqrt(_DIM)).tolist()


class StubEmbedder:
    async def aembed(self, text: str) -> list[float]:
        return _FAKE_EMB

    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        return [_FAKE_EMB for _ in texts]

    def embed_query_sync(self, text: str) -> list[float]:
        return _FAKE_EMB


class StubVectorStore:
    def __init__(self):
        self._docs: list[SearchResult] = []

    async def aadd(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: list[dict],
    ) -> None:
        for id_, text, meta in zip(ids, texts, metadatas):
            self._docs.append(SearchResult(id=id_, text=text, score=0.9, metadata=meta))

    async def asearch(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter: dict | None = None,
    ) -> list[SearchResult]:
        return self._docs[:top_k]

    async def adelete(self, ids: list[str]) -> None:
        self._docs = [d for d in self._docs if d.id not in ids]


class StubGenerator:
    def __init__(self, embedder=None):
        self.embedder = embedder or StubEmbedder()

    async def generate(
        self,
        question: str,
        chunks: list[SearchResult],
        conversation_history: list[dict] | None = None,
    ) -> dict:
        if not chunks:
            return {
                "answer": "I don't have enough information to answer this question based on the provided documents.",
                "citations": [],
                "faithfulness_score": 1.0,
            }
        return {
            "answer": f"Stub answer for: {question}",
            "citations": [
                {
                    "index": 1,
                    "id": chunks[0].id,
                    "filename": chunks[0].metadata.get("filename", "test.txt"),
                    "score": float(chunks[0].score),
                    "text_preview": chunks[0].text[:200],
                }
            ],
            "faithfulness_score": 0.95,
        }


class StubDBSession:
    def add(self, obj: Any) -> None:
        pass

    def add_all(self, objs: list[Any]) -> None:
        pass

    async def flush(self) -> None:
        pass

    async def commit(self) -> None:
        pass

    async def rollback(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def execute(self, *args: Any, **kwargs: Any):
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        result.scalar_one_or_none.return_value = None
        return result

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


_embedder = StubEmbedder()
_vector_store = StubVectorStore()
_generator = StubGenerator(_embedder)
_db = StubDBSession()


@pytest.fixture(autouse=True)
def override_app_dependencies():
    async def _get_db_override():
        yield _db

    app.dependency_overrides[get_db] = _get_db_override
    app.dependency_overrides[get_embedder] = lambda: _embedder
    app.dependency_overrides[get_vector_store] = lambda: _vector_store
    app.dependency_overrides[get_reranker] = lambda: None
    app.dependency_overrides[get_generator] = lambda embedder=None: _generator

    yield

    app.dependency_overrides.clear()
