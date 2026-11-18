"""Hybrid retriever (P3): fuses BM25 and dense rankings with RRF by default."""

from __future__ import annotations

import time

from agentic_rag.domain.retrieval import RetrievalResult, RetrievalStrategy, ScoredChunk
from agentic_rag.retrieval.base import Retriever
from agentic_rag.retrieval.fusion import reciprocal_rank_fusion

# Each stage retrieves more than top_k so the fusion step has material to
# work with; too small a pool here silently caps the hybrid pipeline's recall.
_CANDIDATE_POOL_MULTIPLIER = 4


class HybridRetriever(Retriever):
    """Retrieves from BM25 and dense retrievers independently, then fuses with RRF."""

    def __init__(
        self, sparse_retriever: Retriever, dense_retriever: Retriever, rrf_k: int = 60
    ) -> None:
        """
        Args:
            sparse_retriever: typically a `Bm25Retriever`.
            dense_retriever: typically a `DenseRetriever`.
            rrf_k: the reciprocal rank fusion constant (research doc, H5).
        """
        self._sparse_retriever = sparse_retriever
        self._dense_retriever = dense_retriever
        self._rrf_k = rrf_k

    def retrieve(self, query_text: str, query_id: str, top_k: int) -> RetrievalResult:
        start_time = time.perf_counter()
        candidate_pool_size = top_k * _CANDIDATE_POOL_MULTIPLIER

        sparse_result = self._sparse_retriever.retrieve(query_text, query_id, candidate_pool_size)
        dense_result = self._dense_retriever.retrieve(query_text, query_id, candidate_pool_size)

        fused = reciprocal_rank_fusion(
            [sparse_result.ranked_chunks, dense_result.ranked_chunks], rrf_k=self._rrf_k
        )[:top_k]
        # Re-index after truncation so `rank` stays 0-based and contiguous.
        ranked_chunks = [
            ScoredChunk(chunk=sc.chunk, score=sc.score, rank=i, stage_scores=sc.stage_scores)
            for i, sc in enumerate(fused)
        ]

        return RetrievalResult(
            query_id=query_id,
            strategy=RetrievalStrategy.HYBRID_RRF,
            ranked_chunks=ranked_chunks,
            latency_seconds=time.perf_counter() - start_time,
            trace=[
                f"Sparse candidates: {len(sparse_result.ranked_chunks)}",
                f"Dense candidates: {len(dense_result.ranked_chunks)}",
                f"RRF(k={self._rrf_k}) fused to top {len(ranked_chunks)}",
            ],
        )
