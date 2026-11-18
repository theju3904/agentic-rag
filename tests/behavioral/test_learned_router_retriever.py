"""Behavioral tests for `LearnedRouterRetriever` (the agentic learned_router mode).

These tests use a stub `LearnedRouter` so no trained XGBoost artifact is
required. They assert the orchestration contract: all four sub-pipelines are
run, the strategy on the returned result is `LEARNED_ROUTER`, the chosen
pool's chunks are what gets handed back, and the trace records the decision.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_rag.agent.learned_router_retriever import LearnedRouterRetriever
from agentic_rag.domain.document import Chunk, DatasetSource
from agentic_rag.domain.retrieval import RetrievalResult, RetrievalStrategy, ScoredChunk
from agentic_rag.retrieval.base import Retriever


def _scored_chunk(chunk_id: str, score: float = 1.0) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id=chunk_id,
        source=DatasetSource.SCIFACT,
        text=f"text for {chunk_id}",
        position=0,
    )
    return ScoredChunk(chunk=chunk, score=score, rank=0)


class _StubRetriever(Retriever):
    """Returns a fixed, labelled candidate list regardless of query text."""

    def __init__(self, strategy: RetrievalStrategy, candidates: list[ScoredChunk]) -> None:
        self._strategy = strategy
        self._candidates = candidates
        self.call_count = 0

    def retrieve(self, query_text: str, query_id: str, top_k: int) -> RetrievalResult:
        self.call_count += 1
        return RetrievalResult(
            query_id=query_id,
            strategy=self._strategy,
            ranked_chunks=self._candidates[:top_k],
            latency_seconds=0.0,
            trace=[f"stub {self._strategy.value}"],
        )


@dataclass
class _StubLearnedRouter:
    """Duck-typed stand-in for `LearnedRouter`; returns a preset strategy."""

    choice: RetrievalStrategy
    predict_calls: int = 0

    def predict(
        self,
        query_text: str,
        bm25_doc_ids: list[str],
        dense_doc_ids: list[str],
        rerank_doc_ids: list[str],
        dataset: str | None = None,
    ) -> RetrievalStrategy:
        self.predict_calls += 1
        return self.choice


def _build(choice: RetrievalStrategy) -> tuple[
    LearnedRouterRetriever,
    _StubRetriever,
    _StubRetriever,
    _StubRetriever,
    _StubRetriever,
    _StubLearnedRouter,
]:
    sparse = _StubRetriever(RetrievalStrategy.SPARSE_BM25, [_scored_chunk("bm25-c")])
    dense = _StubRetriever(RetrievalStrategy.DENSE, [_scored_chunk("dense-c")])
    reranked = _StubRetriever(RetrievalStrategy.HYBRID_RERANKED, [_scored_chunk("rerank-c")])
    agentic = _StubRetriever(RetrievalStrategy.AGENTIC_CORRECTIVE, [_scored_chunk("agentic-c")])
    router = _StubLearnedRouter(choice=choice)
    retriever = LearnedRouterRetriever(
        sparse_retriever=sparse,
        dense_retriever=dense,
        reranked_retriever=reranked,
        agentic_retriever=agentic,
        learned_router=router,  # type: ignore[arg-type]  # duck-typed test double
    )
    return retriever, sparse, dense, reranked, agentic, router


class TestLearnedRouterRetriever:
    def test_runs_all_four_sub_retrievers(self) -> None:
        retriever, sparse, dense, reranked, agentic, _ = _build(
            RetrievalStrategy.AGENTIC_CORRECTIVE
        )
        retriever.retrieve("some query", "q1", top_k=10)
        assert sparse.call_count == 1
        assert dense.call_count == 1
        assert reranked.call_count == 1
        assert agentic.call_count == 1

    def test_strategy_on_result_is_learned_router(self) -> None:
        retriever, *_ = _build(RetrievalStrategy.DENSE)
        result = retriever.retrieve("q", "q1", top_k=10)
        assert result.strategy == RetrievalStrategy.LEARNED_ROUTER

    def test_returns_bm25_chunks_when_router_picks_bm25(self) -> None:
        retriever, *_ = _build(RetrievalStrategy.SPARSE_BM25)
        result = retriever.retrieve("q", "q1", top_k=10)
        assert [c.chunk.chunk_id for c in result.ranked_chunks] == ["bm25-c"]
        assert any("Learned router selected pool 'sparse_bm25'" in t for t in result.trace)

    def test_returns_dense_chunks_when_router_picks_dense(self) -> None:
        retriever, *_ = _build(RetrievalStrategy.DENSE)
        result = retriever.retrieve("q", "q1", top_k=10)
        assert [c.chunk.chunk_id for c in result.ranked_chunks] == ["dense-c"]

    def test_returns_reranked_chunks_when_router_picks_hybrid_rerank(self) -> None:
        retriever, *_ = _build(RetrievalStrategy.HYBRID_RERANKED)
        result = retriever.retrieve("q", "q1", top_k=10)
        assert [c.chunk.chunk_id for c in result.ranked_chunks] == ["rerank-c"]

    def test_returns_agentic_chunks_when_router_picks_agentic(self) -> None:
        retriever, *_ = _build(RetrievalStrategy.AGENTIC_CORRECTIVE)
        result = retriever.retrieve("q", "q1", top_k=10)
        assert [c.chunk.chunk_id for c in result.ranked_chunks] == ["agentic-c"]

    def test_falls_back_to_agentic_on_unknown_strategy(self) -> None:
        # HYBRID_RRF is not in the pool list the retriever runs. If a future
        # router emits it, we should degrade to agentic rather than crash.
        retriever, _, _, _, agentic, _ = _build(RetrievalStrategy.HYBRID_RRF)
        result = retriever.retrieve("q", "q1", top_k=10)
        assert [c.chunk.chunk_id for c in result.ranked_chunks] == ["agentic-c"]
        assert any("Unknown routed strategy" in t for t in result.trace)
