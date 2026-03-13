"""
Hybrid Retrieval Engine
=======================
Fuses dense vector search (ANN) with sparse BM25 keyword search via
Reciprocal Rank Fusion (RRF), then reranks the merged list with a
cross-encoder model.

  alpha = 1.0  →  pure dense
  alpha = 0.0  →  pure BM25
  0 < alpha < 1 → hybrid
"""

from __future__ import annotations

import asyncio

import structlog
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.retrieval.embedder import Embedder
from app.retrieval.vector_store import SearchResult, VectorStoreClient

logger = structlog.get_logger(__name__)


# ── BM25 (in-memory, rebuilt per query against retrieved candidates) ──────────
class BM25Engine:
    """Lightweight BM25 over a provided corpus (post-dense-fetch)."""

    def rank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        if not candidates:
            return candidates
        tokenised = [c.text.lower().split() for c in candidates]
        bm25 = BM25Okapi(tokenised)
        scores = bm25.get_scores(query.lower().split())
        for i, c in enumerate(candidates):
            c.metadata["bm25_score"] = float(scores[i])
        return sorted(candidates, key=lambda c: c.metadata["bm25_score"], reverse=True)


# ── Cross-Encoder Reranker ────────────────────────────────────────────────────
class CrossEncoderReranker:
    def __init__(self, model_name: str = settings.RERANKER_MODEL):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        if not candidates:
            return candidates
        pairs = [(query, c.text) for c in candidates]
        scores = self._model.predict(pairs)
        for c, s in zip(candidates, scores):
            c.metadata["rerank_score"] = float(s)
        reranked = sorted(candidates, key=lambda c: c.metadata["rerank_score"], reverse=True)
        logger.debug("reranked", candidates=len(candidates))
        return reranked


# ── RRF fusion ────────────────────────────────────────────────────────────────
def _reciprocal_rank_fusion(
    dense_results: list[SearchResult],
    bm25_results: list[SearchResult],
    alpha: float = settings.HYBRID_ALPHA,
    k: int = 60,
) -> list[SearchResult]:
    """Weighted RRF: score = alpha * (1/dense_rank) + (1-alpha) * (1/bm25_rank)"""
    scores: dict[str, float] = {}
    id_to_result: dict[str, SearchResult] = {}

    for rank, res in enumerate(dense_results):
        scores[res.id] = scores.get(res.id, 0.0) + alpha / (k + rank + 1)
        id_to_result[res.id] = res

    for rank, res in enumerate(bm25_results):
        scores[res.id] = scores.get(res.id, 0.0) + (1 - alpha) / (k + rank + 1)
        id_to_result[res.id] = res

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    fused = []
    for id_ in sorted_ids:
        r = id_to_result[id_]
        r.score = scores[id_]
        fused.append(r)
    return fused


# ── Main Retrieval Engine ─────────────────────────────────────────────────────
class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStoreClient,
        reranker: CrossEncoderReranker | None = None,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25 = BM25Engine()
        self.reranker = reranker if settings.RERANKER_ENABLED else None

    async def retrieve(
        self,
        query: str,
        top_k: int = settings.RETRIEVAL_TOP_K,
        tenant_id: str | None = None,
        alpha: float = settings.HYBRID_ALPHA,
    ) -> list[SearchResult]:
        """
        Full hybrid retrieval pipeline:
        1. Embed query
        2. Dense ANN search (top_k * 3 candidates for fusion)
        3. BM25 re-score within those candidates
        4. RRF fusion
        5. Cross-encoder rerank
        6. Return top_k
        """
        log = logger.bind(query=query[:80], top_k=top_k, tenant_id=tenant_id)

        # 1. Embed
        query_embedding = await self.embedder.aembed(query)

        # 2. Dense search
        dense_filter = {"tenant_id": tenant_id} if tenant_id else None
        dense_results = await self.vector_store.asearch(
            query_embedding=query_embedding,
            top_k=top_k * 3,
            filter=dense_filter,
        )
        log.debug("dense_retrieved", count=len(dense_results))

        if not dense_results:
            return []

        # 3. BM25 within candidates
        bm25_results = self.bm25.rank(query, list(dense_results))

        # 4. RRF fusion
        fused = _reciprocal_rank_fusion(dense_results, bm25_results, alpha=alpha)

        # 5. Rerank
        if self.reranker:
            fused = await asyncio.to_thread(self.reranker.rerank, query, fused[: top_k * 2])

        final = fused[:top_k]
        log.info("retrieval_complete", returned=len(final))
        return final
