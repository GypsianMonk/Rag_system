"""
Unit tests for hybrid retrieval and generation logic.
"""

from unittest.mock import AsyncMock

import pytest

from app.retrieval.retriever import BM25Engine, HybridRetriever, _reciprocal_rank_fusion
from app.retrieval.vector_store import SearchResult


def make_result(id: str, text: str, score: float = 0.8) -> SearchResult:
    return SearchResult(id=id, text=text, score=score, metadata={"filename": f"{id}.pdf"})


@pytest.fixture
def sample_results():
    return [
        make_result("doc1", "Python is a high-level programming language.", 0.9),
        make_result("doc2", "FastAPI is a modern web framework for Python.", 0.8),
        make_result("doc3", "Vector databases store embeddings for similarity search.", 0.7),
    ]


class TestBM25Engine:
    def test_rank_returns_sorted_results(self, sample_results):
        engine = BM25Engine()
        ranked = engine.rank("Python programming language", sample_results)
        assert len(ranked) == 3
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


class TestRRFFusion:
    def test_fusion_combines_results(self, sample_results):
        dense = sample_results[:2]
        bm25 = [sample_results[2], sample_results[0]]
        fused = _reciprocal_rank_fusion(dense, bm25, alpha=0.5)
        assert len(fused) == 3
        ids = [r.id for r in fused]
        assert "doc1" in ids

    def test_pure_dense_alpha_one(self, sample_results):
        fused = _reciprocal_rank_fusion(sample_results, [], alpha=1.0)
        assert fused[0].id == "doc1"

    def test_pure_bm25_alpha_zero(self, sample_results):
        fused = _reciprocal_rank_fusion([], sample_results, alpha=0.0)
        assert fused[0].id == "doc1"


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


class TestFormatContext:
    def test_format_context_builds_string(self):
        from app.generation.generator import _format_context

        chunks = [
            SearchResult(id="c1", text="Python info", score=0.9,
                         metadata={"filename": "python.pdf"}),
            SearchResult(id="c2", text="FastAPI info", score=0.8,
                         metadata={"filename": "fastapi.pdf"}),
        ]
        context_str, citations = _format_context(chunks)
        assert "[Source 1]" in context_str
        assert "[Source 2]" in context_str
        assert len(citations) == 2
        assert citations[0]["index"] == 1
        assert citations[0]["filename"] == "python.pdf"

    def test_format_context_empty(self):
        from app.generation.generator import _format_context

        context_str, citations = _format_context([])
        assert context_str == ""
        assert citations == []

    def test_format_context_uses_source_fallback(self):
        from app.generation.generator import _format_context

        chunks = [SearchResult(id="c1", text="x", score=0.5, metadata={"source": "doc.txt"})]
        _, citations = _format_context(chunks)
        assert citations[0]["filename"] == "doc.txt"

    def test_cosine_similarity_generator(self):
        from app.generation.generator import _cosine_similarity

        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0, abs=1e-9)
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)


class TestDependencies:
    def test_get_retriever_from_stubs(self):
        from app.retrieval.retriever import HybridRetriever
        from app.utils.dependencies import get_retriever
        from tests.conftest import StubEmbedder, StubVectorStore
        retriever = get_retriever(
            embedder=StubEmbedder(),
            vector_store=StubVectorStore(),
            reranker=None,
        )
        assert isinstance(retriever, HybridRetriever)

    def test_optional_tenant_returns_none(self):
        from app.utils.dependencies import optional_tenant
        assert optional_tenant() is None

    def test_optional_tenant_returns_value(self):
        from app.utils.dependencies import optional_tenant
        assert optional_tenant("tenant-1") == "tenant-1"

    def test_get_ingestion_pipeline_returns_pipeline(self):
        from app.ingestion.pipeline import IngestionPipeline
        from app.utils.dependencies import get_ingestion_pipeline
        from tests.conftest import StubDBSession, StubEmbedder, StubVectorStore
        pipeline = get_ingestion_pipeline(
            embedder=StubEmbedder(),
            vector_store=StubVectorStore(),
            db=StubDBSession(),
        )
        assert isinstance(pipeline, IngestionPipeline)
