"""Behavioral tests for the agentic pipeline's corrective decision-making.

These assert properties of the *orchestration logic* (when does it retry?
when does it abstain?) using lightweight fakes for the retrieval/reranking
stages, so they run in milliseconds with no model downloads or database.
"""

from __future__ import annotations

from agentic_rag.agent.agentic_retriever import AgenticRetriever
from agentic_rag.agent.corrective_controller import CorrectiveController
from agentic_rag.agent.query_rewriter import QueryRewriter
from agentic_rag.agent.relevance_grader import RelevanceGrader
from agentic_rag.domain.document import Chunk, DatasetSource
from agentic_rag.domain.retrieval import RetrievalResult, RetrievalStrategy, ScoredChunk
from agentic_rag.generation.llm_client import NullLLMClient
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


class _StubHybridRetriever(Retriever):
    """Always returns the same fixed candidate pool, regardless of query text."""

    def __init__(self, candidates: list[ScoredChunk]) -> None:
        self._candidates = candidates

    def retrieve(self, query_text: str, query_id: str, top_k: int) -> RetrievalResult:
        return RetrievalResult(
            query_id=query_id,
            strategy=RetrievalStrategy.HYBRID_RRF,
            ranked_chunks=self._candidates[:top_k],
            latency_seconds=0.0,
        )


class _StubReranker:
    """Re-scores candidates using a fixed, test-controlled score map (no real model)."""

    def __init__(self, score_by_chunk_id: dict[str, float]) -> None:
        self._score_by_chunk_id = score_by_chunk_id

    def rerank(
        self, query_text: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        rescored = sorted(
            candidates,
            key=lambda sc: self._score_by_chunk_id.get(sc.chunk.chunk_id, sc.score),
            reverse=True,
        )[:top_k]
        return [
            ScoredChunk(
                chunk=sc.chunk,
                score=self._score_by_chunk_id.get(sc.chunk.chunk_id, sc.score),
                rank=i,
            )
            for i, sc in enumerate(rescored)
        ]


def _build_agentic_retriever(
    hybrid_retriever: Retriever,
    reranker: _StubReranker,
    relevant_threshold: float = 0.0,
    irrelevant_threshold: float = -5.0,
) -> AgenticRetriever:
    return AgenticRetriever(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,  # type: ignore[arg-type]  -- duck-typed test double
        relevance_grader=RelevanceGrader(relevant_threshold, irrelevant_threshold),
        query_rewriter=QueryRewriter(NullLLMClient()),
        corrective_controller=CorrectiveController(),
    )


def test_confidently_relevant_pool_proceeds_without_retrying():
    candidates = [_scored_chunk("c1", 10.0)]
    stub_reranker = _StubReranker({"c1": 10.0})
    retriever = _build_agentic_retriever(_StubHybridRetriever(candidates), stub_reranker)

    result = retriever.retrieve("some question", query_id="q1", top_k=5)

    assert len(result.ranked_chunks) == 1
    assert not any("Rewrote query" in line for line in result.trace)


def test_confidently_irrelevant_pool_abstains_with_no_context():
    candidates = [_scored_chunk("c1", -10.0)]
    stub_reranker = _StubReranker({"c1": -10.0})
    retriever = _build_agentic_retriever(_StubHybridRetriever(candidates), stub_reranker)

    result = retriever.retrieve("some question", query_id="q1", top_k=5)

    assert result.ranked_chunks == []
    assert any("ABSTAIN" in line for line in result.trace)


def test_ambiguous_pool_retries_exactly_once_then_abstains_if_still_ambiguous():
    candidates = [_scored_chunk("c1", -2.0)]  # strictly between the two thresholds
    stub_reranker = _StubReranker({"c1": -2.0})
    retriever = _build_agentic_retriever(_StubHybridRetriever(candidates), stub_reranker)

    result = retriever.retrieve("some question", query_id="q1", top_k=5)

    rewrite_events = [line for line in result.trace if "Rewrote query" in line]
    assert len(rewrite_events) == 1  # retried exactly once, never looped
    assert result.ranked_chunks == []  # still ambiguous after the retry, so it abstains


def test_empty_retrieval_pool_abstains_immediately():
    retriever = _build_agentic_retriever(_StubHybridRetriever([]), _StubReranker({}))

    result = retriever.retrieve("some question", query_id="q1", top_k=5)

    assert result.ranked_chunks == []


def test_per_doc_filter_drops_confidently_irrelevant_from_mixed_pool():
    """P5 returns a cleaner top-k than P4 by filtering noisy chunks the reranker ranked high."""
    # Three chunks: one confidently relevant, one ambiguous, one confidently irrelevant.
    candidates = [
        _scored_chunk("c_good", 10.0),
        _scored_chunk("c_mid", -2.0),
        _scored_chunk("c_noise", -10.0),
    ]
    stub_reranker = _StubReranker({"c_good": 10.0, "c_mid": -2.0, "c_noise": -10.0})
    retriever = _build_agentic_retriever(_StubHybridRetriever(candidates), stub_reranker)

    result = retriever.retrieve("some question", query_id="q1", top_k=5)

    returned_ids = [sc.chunk.chunk_id for sc in result.ranked_chunks]
    assert "c_noise" not in returned_ids, "confidently-irrelevant chunk must be filtered out"
    assert "c_good" in returned_ids, "confidently-relevant chunk must survive"
    assert any("Per-doc filter dropped 1/3" in line for line in result.trace)


def test_per_doc_filter_is_no_op_when_all_chunks_pass():
    """When no chunk is confidently irrelevant, P5's output equals P4's (same top-k)."""
    candidates = [_scored_chunk("c1", 10.0), _scored_chunk("c2", -2.0)]
    stub_reranker = _StubReranker({"c1": 10.0, "c2": -2.0})
    retriever = _build_agentic_retriever(_StubHybridRetriever(candidates), stub_reranker)

    result = retriever.retrieve("some question", query_id="q1", top_k=5)

    returned_ids = [sc.chunk.chunk_id for sc in result.ranked_chunks]
    assert returned_ids == ["c1", "c2"], "filter must not drop anything above the threshold"
    assert any("Per-doc filter dropped 0/2" in line for line in result.trace)


def test_per_doc_filter_all_irrelevant_abstains():
    """If every chunk is confidently irrelevant, filter empties the pool and P5 abstains."""
    candidates = [_scored_chunk("c1", -10.0), _scored_chunk("c2", -11.0)]
    stub_reranker = _StubReranker({"c1": -10.0, "c2": -11.0})
    retriever = _build_agentic_retriever(_StubHybridRetriever(candidates), stub_reranker)

    result = retriever.retrieve("some question", query_id="q1", top_k=5)

    assert result.ranked_chunks == []
    assert any("ABSTAIN" in line for line in result.trace)
