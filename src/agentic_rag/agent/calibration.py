"""Calibrates the agentic pipeline's relevance-grading thresholds.

Scores a sample of (query, gold chunk) pairs and (query, random chunk) pairs
with the cross-encoder, then picks thresholds that separate the two score
distributions: the 25th percentile of the "relevant" scores becomes
`confident_relevant_threshold`, and the 75th percentile of the "irrelevant"
scores becomes `confident_irrelevant_threshold`. See README, "Calibrating the
agentic pipeline", for the full methodology and the observed distributions.
"""

from __future__ import annotations

import random

from sentence_transformers import CrossEncoder

from agentic_rag.data.base_loader import LoadedDataset
from agentic_rag.domain.query import Query

# Fallbacks used only if a dataset has too few qrels to sample from at all.
_FALLBACK_RELEVANT_SCORE = 0.0
_FALLBACK_IRRELEVANT_SCORE = -5.0


def calibrate_relevance_thresholds(
    dataset: LoadedDataset,
    cross_encoder_model_name: str,
    sample_size: int,
    random_seed: int,
) -> dict[str, float]:
    """Return `{"confident_relevant_threshold": ..., "confident_irrelevant_threshold": ...}`."""
    rng = random.Random(random_seed)
    text_by_document_id = {document.document_id: document.text for document in dataset.documents}
    query_by_id = {query.query_id: query for query in dataset.queries}

    relevant_pairs, irrelevant_pairs = _sample_pairs(
        dataset, query_by_id, text_by_document_id, sample_size, rng
    )

    model = CrossEncoder(cross_encoder_model_name)
    relevant_scores = (
        sorted(model.predict(relevant_pairs)) if relevant_pairs else [_FALLBACK_RELEVANT_SCORE]
    )
    irrelevant_scores = (
        sorted(model.predict(irrelevant_pairs))
        if irrelevant_pairs
        else [_FALLBACK_IRRELEVANT_SCORE]
    )

    return {
        "confident_relevant_threshold": float(_percentile(relevant_scores, 0.25)),
        "confident_irrelevant_threshold": float(_percentile(irrelevant_scores, 0.75)),
    }


def _sample_pairs(
    dataset: LoadedDataset,
    query_by_id: dict[str, Query],
    text_by_document_id: dict[str, str],
    sample_size: int,
    rng: random.Random,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Build (question, passage) pairs for known-relevant and random passages."""
    document_ids = list(text_by_document_id)
    relevant_document_ids_by_query: dict[str, list[str]] = {}
    for judgment in dataset.judgments:
        if judgment.relevance > 0:
            relevant_document_ids_by_query.setdefault(judgment.query_id, []).append(
                judgment.document_id
            )

    sample_pool = list(relevant_document_ids_by_query)
    sampled_query_ids = rng.sample(sample_pool, min(sample_size, len(sample_pool)))

    relevant_pairs: list[tuple[str, str]] = []
    irrelevant_pairs: list[tuple[str, str]] = []
    for query_id in sampled_query_ids:
        question = query_by_id[query_id].text
        gold_document_id = rng.choice(relevant_document_ids_by_query[query_id])
        relevant_pairs.append((question, text_by_document_id[gold_document_id]))

        random_document_id = rng.choice(document_ids)
        irrelevant_pairs.append((question, text_by_document_id[random_document_id]))
    return relevant_pairs, irrelevant_pairs


def _percentile(sorted_values: list[float], fraction: float) -> float:
    index = int(fraction * (len(sorted_values) - 1))
    return sorted_values[index]
