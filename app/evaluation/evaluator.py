"""
Evaluation Framework
====================
Wraps RAGAS + custom metrics for offline and online evaluation.

Metrics:
  - faithfulness        answer supported by retrieved context
  - context_precision   retrieved chunks contain relevant info
  - context_recall      all ground-truth info present in chunks
  - answer_relevancy    answer addresses the question
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import structlog

from app.core.config import settings
from app.retrieval.embedder import Embedder

logger = structlog.get_logger(__name__)


@dataclass
class EvalSample:
    question: str
    answer: str
    contexts: List[str]                     # retrieved chunks
    ground_truth: Optional[str] = None      # for recall


@dataclass
class EvalResult:
    faithfulness: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    answer_relevancy: float = 0.0
    composite: float = 0.0

    def summary(self) -> dict:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "composite": round(self.composite, 4),
        }


def _cosine(a, b) -> float:
    a, b = np.array(a), np.array(b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0


class RAGEvaluator:
    """
    Embedding-based evaluation (no external API calls required).
    For more advanced RAGAS-based eval, swap with `ragas.evaluate()`.
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    async def evaluate(self, sample: EvalSample) -> EvalResult:
        embs = await self.embedder.aembed_batch(
            [sample.question, sample.answer] + sample.contexts
            + ([sample.ground_truth] if sample.ground_truth else [])
        )
        q_emb = embs[0]
        a_emb = embs[1]
        ctx_embs = embs[2: 2 + len(sample.contexts)]
        gt_emb = embs[-1] if sample.ground_truth else None

        # Faithfulness: avg similarity(answer, context_i)
        faith = float(np.mean([_cosine(a_emb, c) for c in ctx_embs])) if ctx_embs else 0.0

        # Context precision: avg similarity(question, context_i)
        precision = float(np.mean([_cosine(q_emb, c) for c in ctx_embs])) if ctx_embs else 0.0

        # Context recall: similarity(ground_truth, best context) — only if GT given
        recall = float(max([_cosine(gt_emb, c) for c in ctx_embs])) if gt_emb and ctx_embs else 1.0

        # Answer relevancy: similarity(question, answer)
        relevancy = _cosine(q_emb, a_emb)

        composite = float(np.mean([faith, precision, recall, relevancy]))

        return EvalResult(
            faithfulness=faith,
            context_precision=precision,
            context_recall=recall,
            answer_relevancy=relevancy,
            composite=composite,
        )

    async def evaluate_batch(self, samples: List[EvalSample]) -> List[EvalResult]:
        results = await asyncio.gather(*[self.evaluate(s) for s in samples])
        agg = {
            k: round(float(np.mean([getattr(r, k) for r in results])), 4)
            for k in ("faithfulness", "context_precision", "context_recall", "answer_relevancy", "composite")
        }
        logger.info("batch_evaluation_complete", n=len(samples), **agg)
        return results


# ── RAGAS integration (optional, requires `pip install ragas`) ────────────────
async def evaluate_with_ragas(samples: List[EvalSample], llm=None, embeddings=None):
    """
    Use RAGAS library for more rigorous LLM-graded evaluation.
    Requires: pip install ragas
    """
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_eval
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError:
        raise RuntimeError("Install ragas: pip install ragas datasets")

    data = {
        "question": [s.question for s in samples],
        "answer": [s.answer for s in samples],
        "contexts": [s.contexts for s in samples],
        "ground_truth": [s.ground_truth or "" for s in samples],
    }
    dataset = Dataset.from_dict(data)
    result = await asyncio.to_thread(
        ragas_eval,
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )
    return result
