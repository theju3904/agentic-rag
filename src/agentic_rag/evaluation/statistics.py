"""Paired bootstrap statistics for comparing two pipelines' per-query scores.

Follows the research doc's methodology (section 5.4): paired bootstrap
resampling over queries, with Bonferroni correction available when comparing
more than one pair of pipelines at once.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BootstrapComparison:
    """The result of comparing two pipelines' per-query scores via paired bootstrap."""

    mean_difference: float
    confidence_interval_low: float
    confidence_interval_high: float
    p_value: float
    num_resamples: int


def paired_bootstrap_comparison(
    scores_a: list[float],
    scores_b: list[float],
    num_resamples: int = 10_000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> BootstrapComparison:
    """Compare `scores_a` (e.g. the agentic pipeline) against `scores_b` (a baseline).

    Both lists must hold the same per-query metric aligned by query: index
    `i` in `scores_a` and `scores_b` must refer to the same query.

    Raises:
        ValueError: if the two score lists differ in length, or are empty.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must have the same length (paired per query)")
    if not scores_a:
        raise ValueError("cannot bootstrap an empty sample")

    rng = random.Random(random_seed)
    num_queries = len(scores_a)
    observed_difference = _mean(scores_a) - _mean(scores_b)

    resampled_differences = []
    for _ in range(num_resamples):
        sample_indices = [rng.randrange(num_queries) for _ in range(num_queries)]
        resampled_a = [scores_a[i] for i in sample_indices]
        resampled_b = [scores_b[i] for i in sample_indices]
        resampled_differences.append(_mean(resampled_a) - _mean(resampled_b))
    resampled_differences.sort()

    tail_fraction = (1.0 - confidence_level) / 2
    low_index = max(int(tail_fraction * num_resamples), 0)
    high_index = min(int((1.0 - tail_fraction) * num_resamples), num_resamples - 1)

    # Two-sided bootstrap p-value: how often the resampled difference
    # crosses zero relative to the observed direction.
    if observed_difference > 0:
        crossings = sum(1 for diff in resampled_differences if diff <= 0)
    else:
        crossings = sum(1 for diff in resampled_differences if diff >= 0)
    p_value = min(1.0, 2.0 * crossings / num_resamples)

    return BootstrapComparison(
        mean_difference=observed_difference,
        confidence_interval_low=resampled_differences[low_index],
        confidence_interval_high=resampled_differences[high_index],
        p_value=p_value,
        num_resamples=num_resamples,
    )


def bonferroni_correct(p_values: list[float]) -> list[float]:
    """Apply Bonferroni correction for `len(p_values)` simultaneous comparisons."""
    correction_factor = len(p_values)
    return [min(1.0, p * correction_factor) for p in p_values]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)
