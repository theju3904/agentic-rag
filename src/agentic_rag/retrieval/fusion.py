"""Score-fusion algorithms for combining two or more ranked lists into one.

Implements the two fusion strategies from the research doc's ablation plan
(section 5.5): reciprocal rank fusion, which needs no score normalization,
and convex combination, which does, but which can outperform RRF when tuned
(section 4.1, H5).
"""

from __future__ import annotations

from agentic_rag.domain.retrieval import ScoredChunk


def reciprocal_rank_fusion(
    ranked_lists: list[list[ScoredChunk]], rrf_k: int = 60
) -> list[ScoredChunk]:
    """Fuse multiple ranked lists by reciprocal rank (Cormack et al., SIGIR 2009).

    RRF only looks at rank position, not the underlying scores, so it needs no
    score normalization across heterogeneous retrievers (BM25 vs. cosine
    similarity).

    Args:
        ranked_lists: one ranked list of `ScoredChunk` per retriever.
        rrf_k: the fusion constant; higher values flatten the influence of rank.

    Returns:
        A single list of `ScoredChunk` re-ranked by fused score (best first),
        with each chunk's original per-stage scores preserved in `stage_scores`.
    """
    fused_score_by_chunk_id: dict[str, float] = {}
    chunk_by_id: dict[str, object] = {}
    stage_scores_by_chunk_id: dict[str, dict] = {}

    for list_index, ranked_list in enumerate(ranked_lists):
        for scored_chunk in ranked_list:
            chunk_id = scored_chunk.chunk.chunk_id
            fused_score_by_chunk_id[chunk_id] = fused_score_by_chunk_id.get(
                chunk_id, 0.0
            ) + (1.0 / (rrf_k + scored_chunk.rank + 1))
            chunk_by_id[chunk_id] = scored_chunk.chunk
            stage_scores_by_chunk_id.setdefault(chunk_id, {})[
                f"stage_{list_index}"
            ] = scored_chunk.score

    return _to_sorted_scored_chunks(fused_score_by_chunk_id, chunk_by_id, stage_scores_by_chunk_id)


def convex_combination_fusion(
    ranked_lists: list[list[ScoredChunk]], weights: list[float]
) -> list[ScoredChunk]:
    """Fuse multiple ranked lists via a weighted sum of min-max normalized scores.

    Unlike RRF, this uses the actual retrieval scores, so each list is first
    min-max normalized to [0, 1] to make heterogeneous score scales comparable.

    Args:
        ranked_lists: one ranked list of `ScoredChunk` per retriever.
        weights: one weight per list, aligned by index; need not sum to 1.

    Raises:
        ValueError: if `weights` and `ranked_lists` have different lengths.
    """
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must have the same length as ranked_lists")

    fused_score_by_chunk_id: dict[str, float] = {}
    chunk_by_id: dict[str, object] = {}
    stage_scores_by_chunk_id: dict[str, dict] = {}

    for list_index, (ranked_list, weight) in enumerate(zip(ranked_lists, weights, strict=True)):
        normalized_scores = _min_max_normalize([sc.score for sc in ranked_list])
        for scored_chunk, normalized_score in zip(ranked_list, normalized_scores, strict=True):
            chunk_id = scored_chunk.chunk.chunk_id
            fused_score_by_chunk_id[chunk_id] = (
                fused_score_by_chunk_id.get(chunk_id, 0.0) + weight * normalized_score
            )
            chunk_by_id[chunk_id] = scored_chunk.chunk
            stage_scores_by_chunk_id.setdefault(chunk_id, {})[
                f"stage_{list_index}"
            ] = scored_chunk.score

    return _to_sorted_scored_chunks(fused_score_by_chunk_id, chunk_by_id, stage_scores_by_chunk_id)


def _min_max_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lowest, highest = min(scores), max(scores)
    if highest == lowest:
        return [1.0 for _ in scores]  # every candidate tied; treat as equally relevant
    return [(score - lowest) / (highest - lowest) for score in scores]


def _to_sorted_scored_chunks(
    fused_score_by_chunk_id: dict[str, float],
    chunk_by_id: dict,
    stage_scores_by_chunk_id: dict[str, dict],
) -> list[ScoredChunk]:
    ordered_ids = sorted(
        fused_score_by_chunk_id, key=lambda cid: fused_score_by_chunk_id[cid], reverse=True
    )
    return [
        ScoredChunk(
            chunk=chunk_by_id[chunk_id],
            score=fused_score_by_chunk_id[chunk_id],
            rank=rank,
            stage_scores=stage_scores_by_chunk_id[chunk_id],
        )
        for rank, chunk_id in enumerate(ordered_ids)
    ]
