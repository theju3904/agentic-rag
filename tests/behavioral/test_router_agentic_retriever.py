"""Behavioral tests for the `heuristic_router` agentic mode.

Assert that per-query routing goes to the right retriever and that the
abstention path still fires when the chosen pool is confidently irrelevant.
"""

from __future__ import annotations

from agentic_rag.agent.heuristic_query_router import HeuristicQueryRouter
from agentic_rag.agent.relevance_grader import RelevanceGrader
from agentic_rag.agent.router_agentic_retriever import RouterAgenticRetriever
from agentic_rag.domain.document import Chunk, DatasetSource
from agentic_rag.domain.retrieval import RetrievalResult, RetrievalStrategy, ScoredChunk
from agentic_rag.retrieval.base import Retriever


def _scored_chunk(chunk_id: str, score: float) -> ScoredChunk:
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

    def __init__(self, label: str, candidates: list[ScoredChunk]) -> None:
        self._label = label
        self._candidates = candidates
        self.call_count = 0

    def retrieve(self, query_text: str, query_id: str, top_k: int) -> RetrievalResult:
        self.call_count += 1
        return RetrievalResult(
            query_id=query_id,
            strategy=RetrievalStrategy.SPARSE_BM25,  # label irrelevant for this test
            ranked_chunks=self._candidates[:top_k],
            latency_seconds=0.0,
        )


class _StubReranker:
    """No-op reranker that returns inputs unchanged (same order, same scores)."""

    def __init__(self) -> None:
        self.call_count = 0

    def rerank(
        self, query_text: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        self.call_count += 1
        return candidates[:top_k]


def _build(
    sparse: Retriever, hybrid: Retriever, reranker: _StubReranker,
    relevant_t: float = 0.0, irrelevant_t: float = -5.0,
) -> RouterAgenticRetriever:
    return RouterAgenticRetriever(
        sparse_retriever=sparse,
        hybrid_retriever=hybrid,
        reranker=reranker,  # type: ignore[arg-type]  # duck-typed test double
        query_router=HeuristicQueryRouter(),
        relevance_grader=RelevanceGrader(relevant_t, irrelevant_t),
    )


def test_identifier_heavy_query_routes_to_bm25_without_reranking():
    """A query with numbers/years/entities should hit the sparse retriever and skip the reranker."""
    good = [_scored_chunk("c1", 10.0)]
    sparse = _StubRetriever("sparse", good)
    hybrid = _StubRetriever("hybrid", [])
    reranker = _StubReranker()

    agent = _build(sparse, hybrid, reranker)
    result = agent.retrieve("Apple revenue 2024 $100B", "q1", top_k=5)

    assert sparse.call_count == 1
    assert hybrid.call_count == 0
    assert reranker.call_count == 0, "identifier-heavy route must skip rerank entirely"
    assert any("BM25" in line for line in result.trace)
    assert len(result.ranked_chunks) == 1


def test_semantic_query_routes_to_hybrid_plus_rerank():
    """A natural-language why/how question now goes through the hybrid pool + rerank.

    This replaces the earlier dense-only path for the semantic branch:
    hybrid (RRF) dominates dense on every dataset where the reranker helps,
    so there is no reason to feed the reranker a strictly weaker pool.
    """
    good = [_scored_chunk("c1", 10.0)]
    sparse = _StubRetriever("sparse", [])
    hybrid = _StubRetriever("hybrid", good)
    reranker = _StubReranker()

    agent = _build(sparse, hybrid, reranker)
    result = agent.retrieve("why does inflation hurt my savings", "q1", top_k=5)

    assert hybrid.call_count == 1
    assert sparse.call_count == 0
    assert reranker.call_count == 1, "semantic route should rerank"
    assert any("Semantic" in line for line in result.trace), (
        "trace should still label the routing decision as 'Semantic'"
    )


def test_hybrid_fallback_routes_to_hybrid_plus_rerank():
    """A short declarative query with no identifier or semantic signal falls back to hybrid."""
    good = [_scored_chunk("c1", 10.0)]
    sparse = _StubRetriever("sparse", [])
    hybrid = _StubRetriever("hybrid", good)
    reranker = _StubReranker()

    agent = _build(sparse, hybrid, reranker)
    result = agent.retrieve("inflation and interest rates", "q1", top_k=5)

    assert hybrid.call_count == 1
    assert sparse.call_count == 0
    assert reranker.call_count == 1
    assert any("Hybrid" in line for line in result.trace)


def test_abstains_when_chosen_pool_is_confidently_irrelevant():
    """Even in router mode, a bad top-score triggers ABSTAIN (empty return)."""
    noisy = [_scored_chunk("c1", -10.0)]  # below irrelevant threshold
    sparse = _StubRetriever("sparse", noisy)
    hybrid = _StubRetriever("hybrid", [])
    reranker = _StubReranker()

    agent = _build(sparse, hybrid, reranker)
    result = agent.retrieve("Apple 2024 revenue", "q1", top_k=5)

    assert result.ranked_chunks == []
    assert any("ABSTAIN" in line for line in result.trace)
