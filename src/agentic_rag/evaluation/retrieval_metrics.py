"""Retrieval-quality metrics: Recall@k, nDCG@k, MRR@k, MAP@k.

Implemented independently of `pytrec_eval` to avoid another native-extension
dependency. The nDCG formula follows trec_eval's linear-gain definition, so
values may differ from an official BEIR leaderboard submission in the last
decimal place, but are directly, fairly comparable across the five pipelines
in this project, which is what this evaluation is actually for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def recall_at_k(
    retrieved_document_ids: list[str], relevant_document_ids: set[str], k: int
) -> float:
    """Fraction of `relevant_document_ids` that appear in the top-`k` retrieved ids.

    Returns 0.0 (rather than raising) when a query has no relevant documents,
    since that is a labeling artifact, not a retrieval failure to average in.
    """
    if not relevant_document_ids:
        return 0.0
    retrieved_top_k = set(retrieved_document_ids[:k])
    return len(retrieved_top_k & relevant_document_ids) / len(relevant_document_ids)


def reciprocal_rank(
    retrieved_document_ids: list[str], relevant_document_ids: set[str], k: int
) -> float:
    """1/rank of the first relevant document within the top-`k`, or 0.0 if none appear."""
    for rank, document_id in enumerate(retrieved_document_ids[:k], start=1):
        if document_id in relevant_document_ids:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(
    retrieved_document_ids: list[str], relevant_document_ids: set[str], k: int
) -> float:
    """Average precision computed over the top-`k` retrieved ids."""
    if not relevant_document_ids:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, document_id in enumerate(retrieved_document_ids[:k], start=1):
        if document_id in relevant_document_ids:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(len(relevant_document_ids), k)


def ndcg_at_k(
    retrieved_document_ids: list[str], relevance_by_document_id: dict[str, int], k: int
) -> float:
    """Normalized discounted cumulative gain at `k`, using trec_eval's linear-gain formula."""
    if not relevance_by_document_id:
        return 0.0

    dcg = sum(
        relevance_by_document_id.get(document_id, 0) / math.log2(rank + 1)
        for rank, document_id in enumerate(retrieved_document_ids[:k], start=1)
    )
    ideal_gains = sorted(relevance_by_document_id.values(), reverse=True)[:k]
    ideal_dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal_gains, start=1))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


@dataclass(frozen=True, slots=True)
class RetrievalMetricsSummary:
    """Mean retrieval metrics across a set of queries, at a fixed cutoff `k`."""

    recall_at_k: float
    ndcg_at_k: float
    mrr_at_k: float
    map_at_k: float
    num_queries: int


def summarize_retrieval_metrics(
    retrieved_document_ids_by_query: dict[str, list[str]],
    judgments_by_query: dict[str, dict[str, int]],
    k: int,
) -> RetrievalMetricsSummary:
    """Average Recall/nDCG/MRR/MAP@k across every query present in `judgments_by_query`."""
    recalls, ndcgs, mrrs, average_precisions = [], [], [], []
    for query_id, judgments in judgments_by_query.items():
        retrieved = retrieved_document_ids_by_query.get(query_id, [])
        relevant_ids = {document_id for document_id, score in judgments.items() if score > 0}
        recalls.append(recall_at_k(retrieved, relevant_ids, k))
        ndcgs.append(ndcg_at_k(retrieved, judgments, k))
        mrrs.append(reciprocal_rank(retrieved, relevant_ids, k))
        average_precisions.append(average_precision_at_k(retrieved, relevant_ids, k))

    return RetrievalMetricsSummary(
        recall_at_k=_safe_mean(recalls),
        ndcg_at_k=_safe_mean(ndcgs),
        mrr_at_k=_safe_mean(mrrs),
        map_at_k=_safe_mean(average_precisions),
        num_queries=len(recalls),
    )


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
