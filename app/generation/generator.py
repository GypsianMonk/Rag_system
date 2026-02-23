"""
Generation Layer
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np
import structlog
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import settings
from app.retrieval.embedder import Embedder
from app.retrieval.vector_store import SearchResult

logger = structlog.get_logger(__name__)

SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions \
based strictly on the provided context documents.

Rules:
- Base your answer ONLY on the context provided below.
- If the context doesn't contain enough information, say "I don't have enough \
  information to answer this question based on the provided documents."
- Always cite the source document using [Source N] notation where N is the \
  context chunk number.
- Be concise and factual. Do not hallucinate.

Context:
{context}
"""


def _build_llm():
    if settings.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            api_key=settings.OPENAI_API_KEY.get_secret_value(),
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )
    elif settings.LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            api_key=settings.ANTHROPIC_API_KEY.get_secret_value(),
        )
    elif settings.LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=settings.LLM_MODEL, temperature=settings.LLM_TEMPERATURE)
    raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")


def _format_context(chunks: list[SearchResult]) -> tuple[str, list[dict]]:
    context_parts = []
    citations = []
    for i, chunk in enumerate(chunks, start=1):
        filename = chunk.metadata.get("filename", chunk.metadata.get("source", "unknown"))
        context_parts.append(f"[Source {i}] (file: {filename}, score: {chunk.score:.3f})\n{chunk.text}")
        citations.append({
            "index": i,
            "id": chunk.id,
            "filename": filename,
            "score": chunk.score,
            "text_preview": chunk.text[:200],
        })
    return "\n\n---\n\n".join(context_parts), citations


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    return float(np.dot(a_arr, b_arr) / denom) if denom > 0 else 0.0


class RAGGenerator:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self._llm = _build_llm()
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])
        self._chain = self._prompt | self._llm | StrOutputParser()

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

        context, citations = _format_context(chunks)

        history_str = ""
        if conversation_history:
            history_str = "\n".join(
                f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                for m in conversation_history[-6:]
            )
            history_str = f"Conversation History:\n{history_str}\n\n"

        full_question = f"{history_str}Question: {question}"

        answer = await asyncio.to_thread(
            self._chain.invoke,
            {"context": context, "question": full_question},
        )

        faithfulness = await self._compute_faithfulness(answer, chunks)
        if faithfulness < settings.HALLUCINATION_THRESHOLD:
            logger.warning("low_faithfulness_detected", score=faithfulness)

        logger.info("generation_complete", faithfulness=faithfulness, citations=len(citations))
        return {
            "answer": answer,
            "citations": citations,
            "faithfulness_score": round(faithfulness, 4),
        }

    async def astream(self, question: str, chunks: list[SearchResult]) -> AsyncIterator[str]:
        context, _ = _format_context(chunks)
        async for token in self._prompt | self._llm | StrOutputParser():
            yield token

    async def _compute_faithfulness(self, answer: str, chunks: list[SearchResult]) -> float:
        answer_emb = await self.embedder.aembed(answer)
        chunk_embs = await self.embedder.aembed_batch([c.text for c in chunks])
        similarities = [_cosine_similarity(answer_emb, emb) for emb in chunk_embs]
        return float(np.mean(similarities)) if similarities else 0.0
