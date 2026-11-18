"""Tests for numeric-match, exact-match/F1, and abstention-accuracy metrics."""

from __future__ import annotations

from agentic_rag.evaluation.generation_metrics import (
    abstention_accuracy,
    exact_match,
    numeric_match,
    token_f1,
)


def test_numeric_match_accepts_values_within_tolerance():
    assert numeric_match("The answer is 100.5 million.", "100 million")


def test_numeric_match_rejects_values_outside_tolerance():
    assert not numeric_match("The answer is 50.", "100")


def test_numeric_match_handles_thousands_separators():
    assert numeric_match("Revenue was 1,234,000 dollars.", "1234000")


def test_numeric_match_is_false_when_no_number_present():
    assert not numeric_match("no numbers here", "100")


def test_exact_match_ignores_case_and_punctuation():
    assert exact_match("Paris, France!", "paris france")


def test_exact_match_fails_on_different_content():
    assert not exact_match("Paris", "London")


def test_token_f1_partial_overlap_is_between_zero_and_one():
    score = token_f1("the quick brown fox", "the quick fox jumps")
    assert 0.0 < score < 1.0


def test_token_f1_perfect_match_is_one():
    assert token_f1("hello world", "hello world") == 1.0


def test_token_f1_no_overlap_is_zero():
    assert token_f1("hello world", "goodbye moon") == 0.0


def test_abstention_accuracy_counts_matching_decisions():
    predicted = [True, False, True, False]
    gold = [True, False, False, False]
    assert abstention_accuracy(predicted, gold) == 0.75
