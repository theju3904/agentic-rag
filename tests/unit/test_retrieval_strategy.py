"""Pin the top-level vs sub-strategy split so routers and listings stay honest."""

from __future__ import annotations

from agentic_rag.domain.retrieval import RetrievalStrategy


def test_top_level_pipelines_are_the_five_user_facing_ones_in_canonical_order() -> None:
    assert RetrievalStrategy.top_level_pipelines() == (
        RetrievalStrategy.SPARSE_BM25,
        RetrievalStrategy.DENSE,
        RetrievalStrategy.HYBRID_RRF,
        RetrievalStrategy.HYBRID_RERANKED,
        RetrievalStrategy.AGENTIC_CORRECTIVE,
    )


def test_learned_router_is_a_sub_strategy_not_a_top_level_pipeline() -> None:
    assert RetrievalStrategy.LEARNED_ROUTER not in RetrievalStrategy.top_level_pipelines()
    assert RetrievalStrategy.LEARNED_ROUTER.is_top_level_pipeline is False
    for strategy in RetrievalStrategy.top_level_pipelines():
        assert strategy.is_top_level_pipeline is True
