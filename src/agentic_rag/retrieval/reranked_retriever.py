"""Hybrid + cross-encoder reranker retriever (P4)."""

from __future__ import annotations

import time

from agentic_rag.domain.retrieval import RetrievalResult, RetrievalStrategy
from agentic_rag.retrieval.base import Retriever
from agentic_rag.retrieval.cross_encoder_reranker import RerankerProtocol

# H4 in the research doc: reranker recall depends heavily on candidate depth
# (~0.46 Recall@5 at pool=20 vs. ~0.83 at pool=50 in the reference benchmark).
_DEFAULT_CANDIDATE_POOL_SIZE = 100


class RerankedRetriever(Retriever):
    """Retrieves a wide candidate pool from `base_retriever`, then reranks it."""

    def __init__(
        self,
        base_retriever: Retriever,
        reranker: RerankerProtocol,
        candidate_pool_size: int = _DEFAULT_CANDIDATE_POOL_SIZE,
    ) -> None:
        """
        Args:
            base_retriever: usually a `HybridRetriever`; supplies recall.
            reranker: the shared `CrossEncoderReranker`; supplies precision.
            candidate_pool_size: how many candidates `base_retriever` returns
                before reranking (see H4 above -- this is the ceiling on recall).
        """
        self._base_retriever = base_retriever
        self._reranker = reranker
        self._candidate_pool_size = candidate_pool_size

    def retrieve(self, query_text: str, query_id: str, top_k: int) -> RetrievalResult:
        start_time = time.perf_counter()
        base_result = self._base_retriever.retrieve(
            query_text, query_id, self._candidate_pool_size
        )
        ranked_chunks = self._reranker.rerank(query_text, base_result.ranked_chunks, top_k)

        return RetrievalResult(
            query_id=query_id,
            strategy=RetrievalStrategy.HYBRID_RERANKED,
            ranked_chunks=ranked_chunks,
            latency_seconds=time.perf_counter() - start_time,
            trace=[
                *base_result.trace,
                f"Cross-encoder reranked {len(base_result.ranked_chunks)} "
                f"candidates to top {len(ranked_chunks)}",
            ],
        )
