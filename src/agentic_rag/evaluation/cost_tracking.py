"""Cost/latency aggregation across an evaluation run.

LLM dollar cost is computed from token counts using a small price table, so
the numbers stay meaningful if the configured model changes; local/null
backends are treated as zero marginal cost (self-hosted compute cost is out
of scope for this project's cost comparison).
"""

from __future__ import annotations

from dataclasses import dataclass

_PRICE_PER_1K_TOKENS_USD = {
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
}


@dataclass(frozen=True, slots=True)
class CostSummary:
    """Aggregated latency and dollar cost for one pipeline across a run."""

    mean_retrieval_latency_seconds: float
    mean_generation_latency_seconds: float
    total_prompt_tokens: int
    total_completion_tokens: int
    estimated_cost_usd: float


def summarize_cost(
    retrieval_latencies: list[float],
    generation_latencies: list[float],
    prompt_token_counts: list[int],
    completion_token_counts: list[int],
    model_name: str,
) -> CostSummary:
    """Aggregate per-query latency/token lists into one `CostSummary` for `model_name`."""
    prices = _PRICE_PER_1K_TOKENS_USD.get(model_name)
    total_prompt_tokens = sum(prompt_token_counts)
    total_completion_tokens = sum(completion_token_counts)
    estimated_cost = (
        (total_prompt_tokens / 1000) * prices["prompt"]
        + (total_completion_tokens / 1000) * prices["completion"]
        if prices
        else 0.0
    )
    return CostSummary(
        mean_retrieval_latency_seconds=_safe_mean(retrieval_latencies),
        mean_generation_latency_seconds=_safe_mean(generation_latencies),
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        estimated_cost_usd=estimated_cost,
    )


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
