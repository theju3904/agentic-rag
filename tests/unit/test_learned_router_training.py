"""Unit tests for the learned router's training pipeline.

The training script (`agent/learned_router_training.py`) is the code that
produces the artifact behind the project's headline nDCG@10 = 0.521 number.
These tests pin the two pure, isolated pieces of that pipeline — the
tie-aware labelling rule and the ledger-to-sample loader — so a future
refactor cannot silently change what the classifier is trained on.

The XGBoost fit itself is intentionally not unit-tested: training against a
few synthetic samples would exercise the boosted-tree library, not the
project's logic. The `_tie_aware_label` and `_load_training_samples`
functions below cover the project-specific decisions.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pytest

from agentic_rag.agent.learned_router_training import (
    _LEARNED_POOL_STRATEGIES,
    TrainingSample,
    _load_training_samples,
    _tie_aware_label,
)
from agentic_rag.domain.retrieval import RetrievalStrategy

_BM25 = RetrievalStrategy.SPARSE_BM25
_DENSE = RetrievalStrategy.DENSE
_RERANK = RetrievalStrategy.HYBRID_RERANKED
_P5 = RetrievalStrategy.AGENTIC_CORRECTIVE


def _sample(**ndcg: float) -> TrainingSample:
    doc_ids: dict[RetrievalStrategy, list[str]] = {s: [] for s in _LEARNED_POOL_STRATEGIES}
    ndcg_by_strategy = {_BM25: 0.0, _DENSE: 0.0, _RERANK: 0.0, _P5: 0.0}
    ndcg_by_strategy.update({RetrievalStrategy(k): v for k, v in ndcg.items()})
    return TrainingSample(
        query_id="q1",
        query_text="?",
        dataset="scifact",
        doc_ids_by_strategy=doc_ids,
        ndcg_by_strategy=ndcg_by_strategy,
    )


def _p5_index() -> int:
    return _LEARNED_POOL_STRATEGIES.index(_P5)


class TestTieAwareLabel:
    """`_tie_aware_label` is the headline design choice: default to P5 unless
    another pool beats it by a margin wider than `delta`. These tests pin that
    behaviour so a future refactor cannot silently flip to a greedy argmax."""

    def test_defaults_to_p5_when_all_scores_are_identical(self) -> None:
        sample = _sample(sparse_bm25=0.5, dense=0.5, hybrid_reranked=0.5, agentic_corrective=0.5)

        assert _tie_aware_label(sample, delta=0.1) == _p5_index()

    def test_defaults_to_p5_when_another_pool_wins_by_less_than_delta(self) -> None:
        # BM25 is ahead by 0.09 < delta=0.10 -> label stays P5.
        sample = _sample(
            sparse_bm25=0.59, dense=0.3, hybrid_reranked=0.3, agentic_corrective=0.5
        )

        assert _tie_aware_label(sample, delta=0.10) == _p5_index()

    def test_switches_to_the_winner_when_it_beats_p5_by_more_than_delta(self) -> None:
        # BM25 at 0.72, P5 at 0.5 -> margin 0.22 > delta=0.10 -> label becomes BM25.
        sample = _sample(
            sparse_bm25=0.72, dense=0.3, hybrid_reranked=0.3, agentic_corrective=0.5
        )

        label = _tie_aware_label(sample, delta=0.10)

        assert label == _LEARNED_POOL_STRATEGIES.index(_BM25)

    def test_defaults_to_p5_when_p5_is_the_true_best_regardless_of_delta(self) -> None:
        sample = _sample(
            sparse_bm25=0.1, dense=0.1, hybrid_reranked=0.1, agentic_corrective=0.9
        )

        assert _tie_aware_label(sample, delta=0.0) == _p5_index()
        assert _tie_aware_label(sample, delta=0.5) == _p5_index()

    def test_a_tiny_delta_lets_any_clear_winner_override_p5(self) -> None:
        # With delta=0.001, BM25 ahead by 0.01 is enough to flip the label.
        sample = _sample(
            sparse_bm25=0.51, dense=0.3, hybrid_reranked=0.3, agentic_corrective=0.5
        )

        label = _tie_aware_label(sample, delta=0.001)

        assert label == _LEARNED_POOL_STRATEGIES.index(_BM25)

    def test_boundary_at_exactly_delta_still_defaults_to_p5(self) -> None:
        # The check is strict `< delta`, so a margin equal to delta should NOT
        # flip the label. Pins the boundary.
        sample = _sample(
            sparse_bm25=0.60, dense=0.3, hybrid_reranked=0.3, agentic_corrective=0.5
        )

        assert _tie_aware_label(sample, delta=0.10) == _p5_index()


class _FakeConn:
    """Minimal `engine.connect()` context manager returning row iterables."""

    def __init__(self, rows_by_sql_fragment: dict[str, Iterable[tuple[Any, ...]]]) -> None:
        self._rows = rows_by_sql_fragment

    def __enter__(self) -> _FakeConn:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, statement: Any, params: dict[str, Any] | None = None) -> Iterable[tuple]:
        sql = str(statement)
        for fragment, rows in self._rows.items():
            if fragment in sql:
                return list(rows)
        raise AssertionError(f"Unexpected SQL in test: {sql!r}")


class _FakeEngine:
    def __init__(self, rows_by_sql_fragment: dict[str, Iterable[tuple[Any, ...]]]) -> None:
        self._rows = rows_by_sql_fragment

    def connect(self) -> _FakeConn:
        return _FakeConn(self._rows)


class TestLoadTrainingSamples:
    """`_load_training_samples` filters the ledger down to queries that have
    retrievals for every one of the four learned pools AND a relevance
    judgment. These tests pin the two drop conditions so training cannot
    silently include partial rows."""

    def _rows(
        self,
        *,
        judgments: list[tuple[str, str, int]],
        queries: list[tuple[str, str]],
        records: list[tuple[str, str, str, list[str]]],
    ) -> dict[str, list[tuple]]:
        return {
            "FROM relevance_judgments": judgments,
            "FROM queries": queries,
            "FROM pipeline_run_records": records,
        }

    def test_builds_one_sample_per_fully_covered_query(self) -> None:
        engine = _FakeEngine(self._rows(
            judgments=[("q1", "doc_ok", 1)],
            queries=[("q1", "what is X?")],
            records=[
                ("q1", "scifact", s.value, ["doc_ok"]) for s in _LEARNED_POOL_STRATEGIES
            ],
        ))

        samples = _load_training_samples(engine, run_ids=["r"], top_k=10)  # type: ignore[arg-type]

        assert len(samples) == 1
        assert samples[0].query_id == "q1"
        assert samples[0].dataset == "scifact"
        # Every strategy retrieved the one relevant doc, so nDCG@10 is 1.0 for each.
        for strategy in _LEARNED_POOL_STRATEGIES:
            assert samples[0].ndcg_by_strategy[strategy] == pytest.approx(1.0)

    def test_drops_queries_missing_any_of_the_four_strategies(self) -> None:
        # q1 has all four; q2 is missing the agentic row.
        q2_partial = [
            ("q2", "scifact", _BM25.value, ["doc_a"]),
            ("q2", "scifact", _DENSE.value, ["doc_a"]),
            ("q2", "scifact", _RERANK.value, ["doc_a"]),
        ]
        engine = _FakeEngine(self._rows(
            judgments=[("q1", "x", 1), ("q2", "doc_a", 1)],
            queries=[("q1", "?"), ("q2", "?")],
            records=[
                *[("q1", "scifact", s.value, ["x"]) for s in _LEARNED_POOL_STRATEGIES],
                *q2_partial,
            ],
        ))

        samples = _load_training_samples(engine, run_ids=["r"], top_k=10)  # type: ignore[arg-type]

        assert {s.query_id for s in samples} == {"q1"}

    def test_drops_queries_with_no_relevance_judgments(self) -> None:
        engine = _FakeEngine(self._rows(
            judgments=[],  # no qrels for q1 at all
            queries=[("q1", "?")],
            records=[("q1", "scifact", s.value, ["doc_a"]) for s in _LEARNED_POOL_STRATEGIES],
        ))

        samples = _load_training_samples(engine, run_ids=["r"], top_k=10)  # type: ignore[arg-type]

        assert samples == []

    def test_ndcg_is_zero_when_retrieved_misses_the_relevant_doc(self) -> None:
        engine = _FakeEngine(self._rows(
            judgments=[("q1", "doc_relevant", 1)],
            queries=[("q1", "?")],
            records=[
                ("q1", "scifact", s.value, ["doc_other"]) for s in _LEARNED_POOL_STRATEGIES
            ],
        ))

        samples = _load_training_samples(engine, run_ids=["r"], top_k=10)  # type: ignore[arg-type]

        assert len(samples) == 1
        for strategy in _LEARNED_POOL_STRATEGIES:
            assert samples[0].ndcg_by_strategy[strategy] == 0.0
