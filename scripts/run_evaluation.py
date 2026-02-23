#!/usr/bin/env python3
"""
Offline evaluation runner.
Loads a benchmark JSONL dataset and evaluates the RAG pipeline.

Usage:
    python scripts/run_evaluation.py --dataset data/eval_set.jsonl --output results/eval.json
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.evaluation.evaluator import EvalSample, RAGEvaluator
from app.generation.generator import RAGGenerator
from app.retrieval.embedder import Embedder
from app.retrieval.retriever import HybridRetriever
from app.retrieval.vector_store import VectorStoreClient
from app.utils.dependencies import get_embedder, get_vector_store


async def run_eval(dataset_path: str, output_path: str, top_k: int = 5) -> None:
    embedder = get_embedder()
    vector_store = get_vector_store()
    retriever = HybridRetriever(embedder=embedder, vector_store=vector_store)
    generator = RAGGenerator(embedder=embedder)
    evaluator = RAGEvaluator(embedder=embedder)

    samples_data = [json.loads(l) for l in Path(dataset_path).read_text().splitlines() if l.strip()]

    eval_samples: list[EvalSample] = []
    for item in samples_data:
        question = item["question"]
        chunks = await retriever.retrieve(question, top_k=top_k)
        result = await generator.generate(question=question, chunks=chunks)

        eval_samples.append(
            EvalSample(
                question=question,
                answer=result["answer"],
                contexts=[c.text for c in chunks],
                ground_truth=item.get("ground_truth"),
            )
        )

    results = await evaluator.evaluate_batch(eval_samples)
    import numpy as np

    summary = {
        "n_samples": len(results),
        "faithfulness": round(float(np.mean([r.faithfulness for r in results])), 4),
        "context_precision": round(float(np.mean([r.context_precision for r in results])), 4),
        "context_recall": round(float(np.mean([r.context_recall for r in results])), 4),
        "answer_relevancy": round(float(np.mean([r.answer_relevancy for r in results])), 4),
        "composite": round(float(np.mean([r.composite for r in results])), 4),
        "per_sample": [r.summary() for r in results],
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(summary, indent=2))
    print(f"\n{'='*50}\nEvaluation Results\n{'='*50}")
    for k, v in summary.items():
        if k != "per_sample":
            print(f"  {k:<25} {v}")
    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/eval_set.jsonl")
    parser.add_argument("--output", default="results/eval.json")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(run_eval(args.dataset, args.output, args.top_k))
