"""Tests for EvaluationRunner's retrieval-only vs. retrieve+generate passes.

This distinction exists purely for tractability (see README, "Experimental
design"): retrieval metrics never need a generated answer, so the expensive
LLM call must only happen for the smaller subsample passed with
`generate_answers=True`.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from agentic_rag.domain.document import Chunk, DatasetSource
from agentic_rag.domain.evaluation import GenerationResult
from agentic_rag.domain.query import Query, QueryType
from agentic_rag.domain.retrieval import RetrievalResult, RetrievalStrategy, ScoredChunk
from agentic_rag.evaluation.evaluation_runner import EvaluationRunner
from agentic_rag.pipelines.base import RagPipeline


class _FakePipeline(RagPipeline):
    """Records whether `answer` (the generation path) was ever invoked."""

    def __init__(self) -> None:
        self.generation_call_count = 0
        self.strategy = RetrievalStrategy.SPARSE_BM25

    def answer(self, query_id: str, question: str, top_k: int, dataset: str | None = None):
        self.generation_call_count += 1
        generation_result = GenerationResult(
            query_id=query_id,
            strategy=self.strategy,
            answer_text="42",
            abstained=False,
            prompt_tokens=1,
            completion_tokens=1,
            latency_seconds=0.0,
        )
        return self._build_retrieval_result(query_id), generation_result

    def retrieve_only(self, query_id: str, question: str, top_k: int) -> RetrievalResult:
        return self._build_retrieval_result(query_id)

    def _build_retrieval_result(self, query_id: str) -> RetrievalResult:
        chunk = Chunk(
            chunk_id="c1", document_id="d1", source=DatasetSource.SCIFACT, text="x", position=0
        )
        return RetrievalResult(
            query_id=query_id,
            strategy=self.strategy,
            ranked_chunks=[ScoredChunk(chunk=chunk, score=1.0, rank=0)],
            latency_seconds=0.0,
        )


def _query(query_id: str) -> Query:
    return Query(
        query_id=query_id,
        source=DatasetSource.SCIFACT,
        text="q",
        query_type=QueryType.SIMPLE,
        gold_answer="42",
    )


def test_retrieval_only_pass_never_calls_generation():
    pipeline = _FakePipeline()
    runner = EvaluationRunner(MagicMock(), judgments_by_query={})

    reports = runner.run_pass(
        "run1",
        {RetrievalStrategy.SPARSE_BM25: pipeline},
        [_query("q1"), _query("q2")],
        generate_answers=False,
    )

    assert pipeline.generation_call_count == 0
    assert reports[0].mean_answer_score is None


def test_generation_pass_calls_generation_and_scores_answers():
    pipeline = _FakePipeline()
    runner = EvaluationRunner(MagicMock(), judgments_by_query={})

    reports = runner.run_pass(
        "run1", {RetrievalStrategy.SPARSE_BM25: pipeline}, [_query("q1")], generate_answers=True
    )

    assert pipeline.generation_call_count == 1
    assert reports[0].mean_answer_score == 1.0  # predicted "42" exactly matches gold "42"
