"""
/api/v1/query  —  Main RAG query endpoint
"""

import time

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import QueryLog, get_db
from app.generation.generator import RAGGenerator
from app.retrieval.retriever import HybridRetriever
from app.utils.dependencies import get_generator, get_retriever

logger = structlog.get_logger(__name__)
router = APIRouter()


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    top_k: int = Field(default=settings.RETRIEVAL_TOP_K, ge=1, le=20)
    alpha: float = Field(default=settings.HYBRID_ALPHA, ge=0.0, le=1.0)
    tenant_id: str | None = Field(None)
    conversation_history: list[dict] | None = Field(None)
    stream: bool = Field(False)


class Citation(BaseModel):
    index: int
    id: str
    filename: str
    score: float
    text_preview: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    faithfulness_score: float
    latency_ms: float
    tokens_used: int | None = None


@router.post("/query", response_model=QueryResponse, summary="Query the RAG system")
async def query_rag(
    req: QueryRequest,
    db: AsyncSession = Depends(get_db),
    retriever: HybridRetriever = Depends(get_retriever),
    generator: RAGGenerator = Depends(get_generator),
):
    start = time.perf_counter()
    log = logger.bind(query=req.query[:80], tenant_id=req.tenant_id)

    try:
        chunks = await retriever.retrieve(
            query=req.query,
            top_k=req.top_k,
            tenant_id=req.tenant_id,
            alpha=req.alpha,
        )
    except Exception as e:
        log.error("retrieval_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    try:
        result = await generator.generate(
            question=req.query,
            chunks=chunks,
            conversation_history=req.conversation_history,
        )
    except Exception as e:
        log.error("generation_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    query_log = QueryLog(
        tenant_id=req.tenant_id,
        query_text=req.query,
        answer_text=result["answer"],
        retrieved_chunks=[{"id": c["id"], "score": c["score"]} for c in result["citations"]],
        faithfulness_score=result["faithfulness_score"],
        latency_ms=latency_ms,
        model_name=settings.LLM_MODEL,
    )
    db.add(query_log)

    log.info("query_complete", latency_ms=latency_ms, faithfulness=result["faithfulness_score"])
    return QueryResponse(
        query=req.query,
        answer=result["answer"],
        citations=result["citations"],
        faithfulness_score=result["faithfulness_score"],
        latency_ms=latency_ms,
    )
