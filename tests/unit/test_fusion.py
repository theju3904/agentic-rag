"""Tests for the RRF and convex-combination fusion algorithms."""

from __future__ import annotations

import pytest

from agentic_rag.domain.document import Chunk, DatasetSource
from agentic_rag.domain.retrieval import ScoredChunk
from agentic_rag.retrieval.fusion import convex_combination_fusion, reciprocal_rank_fusion


def _scored_chunk(chunk_id: str, score: float, rank: int) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=chunk_id, document_id=chunk_id, source=DatasetSource.SCIFACT, text="x", position=0
    )
    return ScoredChunk(chunk=chunk, score=score, rank=rank)


def test_rrf_promotes_chunks_ranked_highly_in_both_lists():
    list_a = [_scored_chunk("a", 10.0, 0), _scored_chunk("b", 9.0, 1), _scored_chunk("c", 8.0, 2)]
    list_b = [_scored_chunk("b", 0.9, 0), _scored_chunk("a", 0.8, 1), _scored_chunk("c", 0.1, 2)]

    fused_ids = [sc.chunk.chunk_id for sc in reciprocal_rank_fusion([list_a, list_b], rrf_k=60)]

    # "a" and "b" are both in the top two of both lists, so should outrank "c".
    assert fused_ids.index("c") == 2
    assert set(fused_ids[:2]) == {"a", "b"}


def test_rrf_preserves_stage_scores_per_source_list():
    fused = reciprocal_rank_fusion([[_scored_chunk("a", 10.0, 0)], [_scored_chunk("a", 0.5, 0)]])
    assert fused[0].stage_scores == {"stage_0": 10.0, "stage_1": 0.5}


def test_convex_combination_requires_matching_weight_count():
    with pytest.raises(ValueError):
        convex_combination_fusion([[_scored_chunk("a", 1.0, 0)]], weights=[0.5, 0.5])


def test_convex_combination_weights_favor_the_higher_weighted_list():
    list_a = [_scored_chunk("a", 1.0, 0), _scored_chunk("b", 0.0, 1)]
    list_b = [_scored_chunk("b", 1.0, 0), _scored_chunk("a", 0.0, 1)]

    fused = convex_combination_fusion([list_a, list_b], weights=[0.9, 0.1])
    assert fused[0].chunk.chunk_id == "a"
