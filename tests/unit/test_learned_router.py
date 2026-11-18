"""Unit tests for the learned router's (`LEARNED_ROUTER`) feature extraction.

These tests cover the free functions in `agent/learned_router.py` so the
feature contract that training depends on is pinned without needing a trained
XGBoost artifact.
"""

from __future__ import annotations

from agentic_rag.agent.learned_router import (
    OVERLAP_FEATURE_NAMES,
    TEXT_FEATURE_NAMES,
    extract_overlap_features,
    extract_text_features,
)


class TestTextFeatures:
    def test_all_declared_names_are_returned(self) -> None:
        feats = extract_text_features("Apple reported $1.2B in 2024.")
        assert set(feats.keys()) == set(TEXT_FEATURE_NAMES)

    def test_year_and_money_detected(self) -> None:
        feats = extract_text_features("Apple reported $1.2B in 2024.")
        assert feats["has_year"] == 1.0
        assert feats["has_money"] == 1.0
        assert feats["num_digits"] >= 5

    def test_question_mark_and_whword(self) -> None:
        feats = extract_text_features("What is the capital of France?")
        assert feats["has_question_word"] == 1.0
        assert feats["ends_with_qmark"] == 1.0

    def test_leading_capital_not_double_counted(self) -> None:
        # "What" leads the sentence; the sentence-leading capital rule mirrors
        # HeuristicQueryRouter and must not inflate pct_capital_tokens.
        feats = extract_text_features("What causes diabetes")
        assert feats["pct_capital_tokens"] == 0.0

    def test_quoted_strings(self) -> None:
        feats = extract_text_features('search for "exact phrase" please')
        assert feats["num_quoted_strings"] == 1.0

    def test_empty_query_is_handled(self) -> None:
        feats = extract_text_features("")
        assert feats["len_tokens"] == 0.0
        assert feats["len_chars"] == 0.0
        assert feats["pct_capital_tokens"] == 0.0


class TestOverlapFeatures:
    def test_all_declared_names_are_returned(self) -> None:
        feats = extract_overlap_features(["a"], ["b"], ["c"])
        assert set(feats.keys()) == set(OVERLAP_FEATURE_NAMES)

    def test_top1_same_when_bm25_and_dense_agree(self) -> None:
        feats = extract_overlap_features(["x", "y"], ["x", "z"], ["q"])
        assert feats["bm25_dense_top1_same"] == 1.0

    def test_top1_same_when_bm25_matches_rerank(self) -> None:
        feats = extract_overlap_features(["x"], ["y"], ["x"])
        assert feats["bm25_rerank_top1_same"] == 1.0

    def test_jaccard_disjoint_is_zero(self) -> None:
        feats = extract_overlap_features(["a", "b", "c"], ["x", "y", "z"], [])
        assert feats["bm25_dense_jaccard_top5"] == 0.0
        assert feats["bm25_dense_jaccard_top10"] == 0.0

    def test_jaccard_identical_is_one(self) -> None:
        same = ["a", "b", "c", "d", "e"]
        feats = extract_overlap_features(same, same, [])
        assert feats["bm25_dense_jaccard_top5"] == 1.0

    def test_empty_lists_do_not_crash(self) -> None:
        feats = extract_overlap_features([], [], [])
        assert feats["bm25_dense_top1_same"] == 0.0
        assert feats["bm25_rerank_top1_same"] == 0.0
