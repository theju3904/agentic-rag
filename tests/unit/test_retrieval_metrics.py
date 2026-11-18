"""Tests for Recall/nDCG/MRR/MAP@k against small, hand-checked examples."""

from __future__ import annotations

from pytest import approx

from agentic_rag.evaluation.retrieval_metrics import (
    average_precision_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    summarize_retrieval_metrics,
)


def test_recall_at_k_counts_relevant_documents_in_top_k():
    retrieved = ["d1", "d2", "d3", "d4"]
    relevant = {"d2", "d4", "d5"}
    assert recall_at_k(retrieved, relevant, k=4) == approx(2 / 3)


def test_recall_at_k_returns_zero_when_no_relevant_documents_exist():
    assert recall_at_k(["d1"], set(), k=10) == 0.0


def test_reciprocal_rank_finds_first_relevant_hit():
    assert reciprocal_rank(["d1", "d2", "d3"], {"d2"}, k=3) == approx(0.5)


def test_reciprocal_rank_is_zero_when_nothing_relevant_is_retrieved():
    assert reciprocal_rank(["d1", "d2"], {"d9"}, k=2) == 0.0


def test_average_precision_rewards_relevant_documents_ranked_earlier():
    relevant = {"d1"}
    early = average_precision_at_k(["d1", "d2", "d3"], relevant, k=3)
    late = average_precision_at_k(["d3", "d2", "d1"], relevant, k=3)
    assert early > late


def test_ndcg_is_one_for_a_perfectly_ordered_ranking():
    relevance = {"d1": 2, "d2": 1, "d3": 0}
    assert ndcg_at_k(["d1", "d2", "d3"], relevance, k=3) == approx(1.0)


def test_ndcg_is_zero_with_no_relevance_information():
    assert ndcg_at_k(["d1"], {}, k=1) == 0.0


def test_summarize_retrieval_metrics_averages_across_queries():
    retrieved_by_query = {"q1": ["d1", "d2"], "q2": ["d9", "d8"]}
    judgments_by_query = {"q1": {"d1": 1}, "q2": {"d7": 1}}

    summary = summarize_retrieval_metrics(retrieved_by_query, judgments_by_query, k=2)

    assert summary.num_queries == 2
    assert summary.recall_at_k == approx(0.5)  # q1 is a perfect hit, q2 a total miss
