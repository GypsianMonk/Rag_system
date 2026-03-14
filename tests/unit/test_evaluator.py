"""
Unit tests for RAGEvaluator and EvalSample/EvalResult dataclasses.
"""

from __future__ import annotations

import pytest

from app.evaluation.evaluator import EvalResult, EvalSample, RAGEvaluator, _cosine
from tests.conftest import StubEmbedder


class TestCosine:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert _cosine(v, v) == pytest.approx(1.0, abs=1e-9)

    def test_orthogonal_vectors(self):
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self):
        assert _cosine([0.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)


class TestEvalResult:
    def test_summary_keys(self):
        r = EvalResult(
            faithfulness=0.9,
            context_precision=0.8,
            context_recall=0.7,
            answer_relevancy=0.85,
            composite=0.81,
        )
        s = r.summary()
        assert set(s.keys()) == {
            "faithfulness",
            "context_precision",
            "context_recall",
            "answer_relevancy",
            "composite",
        }

    def test_summary_rounds_to_4dp(self):
        r = EvalResult(faithfulness=0.123456)
        assert r.summary()["faithfulness"] == 0.1235

    def test_default_values_are_zero(self):
        r = EvalResult()
        assert r.faithfulness == 0.0
        assert r.composite == 0.0


class TestRAGEvaluator:
    @pytest.fixture
    def evaluator(self):
        return RAGEvaluator(embedder=StubEmbedder())

    @pytest.mark.asyncio
    async def test_evaluate_returns_eval_result(self, evaluator):
        sample = EvalSample(
            question="What is Python?",
            answer="Python is a programming language.",
            contexts=["Python is a high-level language.", "It was created by Guido."],
        )
        result = await evaluator.evaluate(sample)
        assert isinstance(result, EvalResult)
        assert result.faithfulness == pytest.approx(1.0, abs=1e-9)
        assert result.composite == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.asyncio
    async def test_evaluate_with_ground_truth(self, evaluator):
        sample = EvalSample(
            question="What is Python?",
            answer="Python is a programming language.",
            contexts=["Python is a high-level language."],
            ground_truth="Python is a programming language used for many tasks.",
        )
        result = await evaluator.evaluate(sample)
        assert isinstance(result, EvalResult)
        assert result.context_recall == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.asyncio
    async def test_evaluate_no_contexts(self, evaluator):
        sample = EvalSample(question="What?", answer="Something.", contexts=[])
        result = await evaluator.evaluate(sample)
        assert result.faithfulness == 0.0
        assert result.context_precision == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_no_ground_truth_recall_is_one(self, evaluator):
        sample = EvalSample(question="Q", answer="A", contexts=["C"], ground_truth=None)
        result = await evaluator.evaluate(sample)
        assert result.context_recall == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_batch(self, evaluator):
        samples = [
            EvalSample(question=f"Q{i}", answer=f"A{i}", contexts=[f"C{i}"]) for i in range(3)
        ]
        results = await evaluator.evaluate_batch(samples)
        assert len(results) == 3
        assert all(isinstance(r, EvalResult) for r in results)

    @pytest.mark.asyncio
    async def test_composite_is_mean_of_components(self, evaluator):
        sample = EvalSample(question="Q", answer="A", contexts=["C"])
        result = await evaluator.evaluate(sample)
        expected = (
            result.faithfulness
            + result.context_precision
            + result.context_recall
            + result.answer_relevancy
        ) / 4
        assert result.composite == pytest.approx(expected, abs=1e-4)
