"""The learned-router mode of the agentic pipeline: run the candidate pools, let XGBoost pick.

This is the retriever behind `AGENTIC_MODE=learned_router` (the agentic
pipeline's recommended default). The agentic sub-strategy `LEARNED_ROUTER`
(historically "P6") in `RetrievalStrategy` names this same retriever for the
training ledger and historical queries; both references share this
implementation.

Behavior
--------
For each query, this retriever runs the four candidate pools whose scores the
router was trained to arbitrate between:

  BM25            -> P1 top-k
  Dense           -> P2 top-k
  Hybrid + rerank -> P4 top-k (used for the `bm25_rerank_top1_same` feature too)
  Agentic         -> heuristic-router P5 (the "default" branch; wins the tie-aware target)

It then computes 12 text features + 4 cheap retrieval-overlap features over the
BM25 / Dense / Reranked doc-id lists and asks the trained `LearnedRouter` which
pool to hand back. The strategy on the returned `RetrievalResult` is always
`LEARNED_ROUTER`, with the actually-picked pool recorded in `trace` so
downstream reporting can see what the router chose per query.

Latency
-------
Every query pays for all four candidate pipelines - about P1 + P2 + P4 + P5
latency summed. This is the honest cost of a learned router: the training
features depend on three different retrieval outputs, so there is no "free"
early-exit without sacrificing accuracy.
"""

from __future__ import annotations

import time

from agentic_rag.agent.learned_router import LearnedRouter
from agentic_rag.domain.retrieval import RetrievalResult, RetrievalStrategy
from agentic_rag.retrieval.base import Retriever


class LearnedRouterRetriever(Retriever):
    """Runs BM25 / Dense / Hybrid+Rerank / heuristic-router P5; a trained XGBoost picks a pool."""

    def __init__(
        self,
        sparse_retriever: Retriever,
        dense_retriever: Retriever,
        reranked_retriever: Retriever,
        agentic_retriever: Retriever,
        learned_router: LearnedRouter,
    ) -> None:
        self._sparse_retriever = sparse_retriever
        self._dense_retriever = dense_retriever
        self._reranked_retriever = reranked_retriever
        self._agentic_retriever = agentic_retriever
        self._learned_router = learned_router

    def retrieve(self, query_text: str, query_id: str, top_k: int) -> RetrievalResult:
        start_time = time.perf_counter()
        trace: list[str] = []

        bm25_res = self._sparse_retriever.retrieve(query_text, query_id, top_k)
        dense_res = self._dense_retriever.retrieve(query_text, query_id, top_k)
        rerank_res = self._reranked_retriever.retrieve(query_text, query_id, top_k)
        agentic_res = self._agentic_retriever.retrieve(query_text, query_id, top_k)

        bm25_doc_ids = bm25_res.top_k_document_ids(top_k)
        dense_doc_ids = dense_res.top_k_document_ids(top_k)
        rerank_doc_ids = rerank_res.top_k_document_ids(top_k)

        chosen = self._learned_router.predict(
            query_text=query_text,
            bm25_doc_ids=bm25_doc_ids,
            dense_doc_ids=dense_doc_ids,
            rerank_doc_ids=rerank_doc_ids,
        )
        trace.append(f"Learned router selected pool '{chosen.value}'")

        if chosen == RetrievalStrategy.SPARSE_BM25:
            final = bm25_res
        elif chosen == RetrievalStrategy.DENSE:
            final = dense_res
        elif chosen == RetrievalStrategy.HYBRID_RERANKED:
            final = rerank_res
        elif chosen == RetrievalStrategy.AGENTIC_CORRECTIVE:
            final = agentic_res
        else:
            # Future-proofing: if a trained model predicts a strategy we did
            # not pre-run (e.g. HYBRID_RRF added later), fall back to agentic.
            trace.append(f"Unknown routed strategy '{chosen.value}'; falling back to agentic")
            final = agentic_res

        return RetrievalResult(
            query_id=query_id,
            strategy=RetrievalStrategy.LEARNED_ROUTER,
            ranked_chunks=final.ranked_chunks,
            latency_seconds=time.perf_counter() - start_time,
            trace=trace + list(final.trace),
        )
