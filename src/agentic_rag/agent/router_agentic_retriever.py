"""`heuristic_router` mode of the agentic pipeline.

Alternative to the default per-doc-filter `AgenticRetriever`. This variant
classifies each query via `HeuristicQueryRouter` and routes it to the retriever
best-suited to that query shape:

  identifier_heavy -> BM25 top-k, NO rerank
  semantic         -> Hybrid (BM25+Dense RRF) top-N, then cross-encoder rerank
  hybrid           -> Hybrid (BM25+Dense RRF) top-N, then cross-encoder rerank

The semantic and hybrid branches intentionally share a retrieval path. In the
full-set evaluation hybrid (RRF) strictly dominates dense-only on every
dataset where the reranker is useful (SciFact, NFCorpus, FiQA), so feeding the
reranker a hybrid pool instead of a dense-only pool is a strict improvement
for the semantic branch. The two branches are kept distinct in the trace so
the routing decision stays visible even where the retrieval path is the same.

The grader still runs on the chosen pool so the pipeline can abstain when the
top result is confidently irrelevant - this preserves the agentic safety
property regardless of which retriever was chosen.

Motivation and expected gain are documented in
`report/RESULTS_TECHNICAL.md` and `code_understanding.md` section 26.
"""

from __future__ import annotations

import time

from agentic_rag.agent.corrective_controller import CorrectiveAction, CorrectiveController
from agentic_rag.agent.heuristic_query_router import HeuristicQueryRouter, QueryRoute
from agentic_rag.agent.relevance_grader import RelevanceGrader
from agentic_rag.domain.retrieval import RetrievalResult, RetrievalStrategy, ScoredChunk
from agentic_rag.retrieval.base import Retriever
from agentic_rag.retrieval.cross_encoder_reranker import RerankerProtocol

_CANDIDATE_POOL_SIZE = 100


class RouterAgenticRetriever(Retriever):
    """Agentic retriever variant that routes per query (identifier/semantic/hybrid).

    Implements the plain `Retriever` interface so `PipelineFactory` and
    `EvaluationRunner` treat it exactly like any other retriever.
    """

    def __init__(
        self,
        sparse_retriever: Retriever,
        hybrid_retriever: Retriever,
        reranker: RerankerProtocol,
        query_router: HeuristicQueryRouter,
        relevance_grader: RelevanceGrader,
        candidate_pool_size: int = _CANDIDATE_POOL_SIZE,
        corrective_controller: CorrectiveController | None = None,
    ) -> None:
        self._sparse_retriever = sparse_retriever
        self._hybrid_retriever = hybrid_retriever
        self._reranker = reranker
        self._query_router = query_router
        self._relevance_grader = relevance_grader
        self._candidate_pool_size = candidate_pool_size
        # Controller kept for parity with the per_doc variant, though this mode
        # does not do query rewrites - it stays as proceed/abstain only.
        self._corrective_controller = corrective_controller or CorrectiveController()

    def retrieve(self, query_text: str, query_id: str, top_k: int) -> RetrievalResult:
        start_time = time.perf_counter()
        trace: list[str] = []

        route = self._query_router.route(query_text)
        trace.append(f"Router classified query as '{route.value}'")

        if route == QueryRoute.IDENTIFIER_HEAVY:
            # Raw BM25 top-k, NO rerank. The full-set evidence shows reranking
            # HURTS on identifier-heavy corpora (MultiHop, T2), so skipping it
            # is intentional and the point of this mode.
            base = self._sparse_retriever.retrieve(query_text, query_id, top_k)
            candidates = base.ranked_chunks
            trace.append(f"Took BM25 top-{top_k}; skipped rerank")

        else:  # SEMANTIC or HYBRID - both now use the hybrid pool before rerank.
            # Hybrid RRF strictly dominates dense-only on every dataset where
            # the reranker helps (SciFact, NFCorpus, FiQA), so running semantic
            # queries through dense-only cost ranking quality for no benefit.
            base = self._hybrid_retriever.retrieve(
                query_text, query_id, self._candidate_pool_size
            )
            candidates = self._reranker.rerank(query_text, base.ranked_chunks, top_k)
            label = "Semantic" if route == QueryRoute.SEMANTIC else "Hybrid"
            trace.append(
                f"{label} branch: hybrid top-{self._candidate_pool_size} + rerank "
                f"to top-{top_k}"
            )

        # Safety: grade the pool and abstain if confidently irrelevant. No
        # retry loop here - the router makes a single routing decision and the
        # abstention path handles "we have nothing useful."
        grade = self._relevance_grader.grade(candidates)
        trace.append(
            f"Grade: {grade.grade.value} (top_score={grade.top_score:.3f})"
        )
        action = self._corrective_controller.decide(grade.grade, already_retried=True)
        if action == CorrectiveAction.ABSTAIN:
            trace.append("Controller decided to ABSTAIN: returning no context")
            final_chunks: list[ScoredChunk] = []
        else:
            final_chunks = candidates

        return RetrievalResult(
            query_id=query_id,
            strategy=RetrievalStrategy.AGENTIC_CORRECTIVE,
            ranked_chunks=final_chunks,
            latency_seconds=time.perf_counter() - start_time,
            trace=trace,
        )
