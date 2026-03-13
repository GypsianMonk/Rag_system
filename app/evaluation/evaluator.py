"""
Evaluation Framework
"""

import asyncio
from dataclasses import dataclass

import numpy as np
import structlog

from app.retrieval.embedder import Embedder

logger = structlog.get_logger(__name__)


@dataclass
class EvalSample:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str | None = None


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
    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    async def evaluate(self, sample: EvalSample) -> EvalResult:
        embs = await self.embedder.aembed_batch(
            [sample.question, sample.answer]
            + sample.contexts
            + ([sample.ground_truth] if sample.ground_truth else [])
        )
        q_emb = embs[0]
        a_emb = embs[1]
        ctx_embs = embs[2 : 2 + len(sample.contexts)]
        gt_emb = embs[-1] if sample.ground_truth else None

        faith = float(np.mean([_cosine(a_emb, c) for c in ctx_embs])) if ctx_embs else 0.0
        precision = float(np.mean([_cosine(q_emb, c) for c in ctx_embs])) if ctx_embs else 0.0
        recall = float(max([_cosine(gt_emb, c) for c in ctx_embs])) if gt_emb and ctx_embs else 1.0
        relevancy = _cosine(q_emb, a_emb)
        composite = float(np.mean([faith, precision, recall, relevancy]))

        return EvalResult(
            faithfulness=faith,
            context_precision=precision,
            context_recall=recall,
            answer_relevancy=relevancy,
            composite=composite,
        )

    async def evaluate_batch(self, samples: list[EvalSample]) -> list[EvalResult]:
        results = await asyncio.gather(*[self.evaluate(s) for s in samples])
        agg = {
            k: round(float(np.mean([getattr(r, k) for r in results])), 4)
            for k in (
                "faithfulness",
                "context_precision",
                "context_recall",
                "answer_relevancy",
                "composite",
            )
        }
        logger.info("batch_evaluation_complete", n=len(samples), **agg)
        return results


async def evaluate_with_ragas(samples: list[EvalSample], llm=None, embeddings=None):
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
