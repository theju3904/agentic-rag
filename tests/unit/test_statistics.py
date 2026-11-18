"""Tests for the paired bootstrap comparison used to compare pipelines."""

from __future__ import annotations

import random

import pytest

from agentic_rag.evaluation.statistics import bonferroni_correct, paired_bootstrap_comparison


def test_identical_score_lists_have_zero_mean_difference():
    scores = [0.5, 0.6, 0.7, 0.8, 0.9] * 5
    result = paired_bootstrap_comparison(scores, scores, num_resamples=500)
    assert result.mean_difference == 0.0


def test_clearly_better_pipeline_yields_a_small_p_value():
    better, worse = [1.0] * 40, [0.0] * 40
    result = paired_bootstrap_comparison(better, worse, num_resamples=1000)
    assert result.mean_difference == 1.0
    assert result.p_value < 0.05


def test_mixed_scores_produce_a_confidence_interval_containing_the_observed_difference():
    rng = random.Random(0)
    better = [1.0 if rng.random() < 0.7 else 0.0 for _ in range(60)]
    worse = [1.0 if rng.random() < 0.3 else 0.0 for _ in range(60)]

    result = paired_bootstrap_comparison(better, worse, num_resamples=2000, random_seed=1)

    assert result.confidence_interval_low <= result.mean_difference
    assert result.mean_difference <= result.confidence_interval_high


def test_mismatched_lengths_raise_value_error():
    with pytest.raises(ValueError):
        paired_bootstrap_comparison([1.0], [1.0, 2.0])


def test_empty_lists_raise_value_error():
    with pytest.raises(ValueError):
        paired_bootstrap_comparison([], [])


def test_bonferroni_correction_scales_p_values():
    assert bonferroni_correct([0.01, 0.2, 0.5]) == pytest.approx([0.03, 0.6, 1.0])
