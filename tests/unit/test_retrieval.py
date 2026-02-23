"""
Unit tests for hybrid retrieval and generation logic.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.retrieval.retriever import BM25Engine, HybridRetriever, _reciprocal_rank_fusion
from app.retrieval.vector_store import SearchResult


# ── Fixtures ──────────────────────────────────────────────────────────────────
def make_result(id: str, text: str, score: float = 0.8) -> SearchResult:
    return SearchResult(id=id, text=text, score=score, metadata={"filename": f"{id}.pdf"})


@pytest.fixture
def sample_results():
    return [
        make_result("doc1", "Python is a high-level programming language.", 0.9),
        make_result("doc2", "FastAPI is a modern web framework for Python.", 0.8),
        make_result("doc3", "Vector databases store embeddings for similarity search.", 0.7),
    ]


# ── BM25 ──────────────────────────────────────────────────────────────────────
class TestBM25Engine:
    def test_rank_returns_sorted_results(self, sample_results):
        engine = BM25Engine()
        ranked = engine.rank("Python programming language", sample_results)
        assert len(ranked) == 3
        # Python-related docs should rank higher
        assert ranked[0].id in ("doc1", "doc2")

    def test_rank_empty_candidates(self):
        engine = BM25Engine()
        result = engine.rank("anything", [])
        assert result == []

    def test_bm25_score_added_to_metadata(self, sample_results):
        engine = BM25Engine()
        ranked = engine.rank("python", sample_results)
        for r in ranked:
            assert "bm25_score" in r.metadata


# ── RRF Fusion ───────────────────────────────────────────────────────────────
class TestRRFFusion:
    def test_fusion_combines_results(self, sample_results):
        dense = sample_results[:2]
        bm25 = [sample_results[2], sample_results[0]]
        fused = _reciprocal_rank_fusion(dense, bm25, alpha=0.5)
        assert len(fused) == 3
        ids = [r.id for r in fused]
        assert "doc1" in ids  # appears in both, should rank high

    def test_pure_dense_alpha_one(self, sample_results):
        fused = _reciprocal_rank_fusion(sample_results, [], alpha=1.0)
        assert fused[0].id == "doc1"

    def test_pure_bm25_alpha_zero(self, sample_results):
        fused = _reciprocal_rank_fusion([], sample_results, alpha=0.0)
        assert fused[0].id == "doc1"


# ── HybridRetriever ───────────────────────────────────────────────────────────
class TestHybridRetriever:
    @pytest.fixture
    def mock_embedder(self):
        embedder = AsyncMock()
        embedder.aembed.return_value = [0.1] * 768
        return embedder

    @pytest.fixture
    def mock_vector_store(self, sample_results):
        vs = AsyncMock()
        vs.asearch.return_value = sample_results
        return vs

    @pytest.mark.asyncio
    async def test_retrieve_returns_results(self, mock_embedder, mock_vector_store):
        retriever = HybridRetriever(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            reranker=None,
        )
        results = await retriever.retrieve("python web framework", top_k=2)
        assert len(results) <= 2
        mock_embedder.aembed.assert_awaited_once()
        mock_vector_store.asearch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retrieve_empty_vector_store(self, mock_embedder):
        vs = AsyncMock()
        vs.asearch.return_value = []
        retriever = HybridRetriever(embedder=mock_embedder, vector_store=vs, reranker=None)
        results = await retriever.retrieve("anything", top_k=5)
        assert results == []


# ── Generation ────────────────────────────────────────────────────────────────
class TestRAGGenerator:
    @pytest.mark.asyncio
    async def test_generate_no_chunks_returns_fallback(self):
        from app.generation.generator import RAGGenerator
        mock_embedder = AsyncMock()
        gen = RAGGenerator.__new__(RAGGenerator)
        gen.embedder = mock_embedder

        result = await gen.generate("What is Python?", chunks=[])
        assert "don't have enough information" in result["answer"]
        assert result["citations"] == []
        assert result["faithfulness_score"] == 1.0

    @pytest.mark.asyncio
    async def test_cosine_similarity_identical_vectors(self):
        from app.generation.generator import _cosine_similarity
        v = [1.0, 0.0, 0.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_cosine_similarity_orthogonal_vectors(self):
        from app.generation.generator import _cosine_similarity
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
