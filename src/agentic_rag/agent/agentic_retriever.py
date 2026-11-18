"""P5: the agentic corrective RAG retriever, orchestrated as a LangGraph `StateGraph`.

Composes retrieval (a `HybridRetriever`), precision (`CrossEncoderReranker`),
a relevance check (`RelevanceGrader`), and a corrective decision
(`CorrectiveController`) into one retriever that can proceed, rewrite-and-retry,
or abstain per query. The graph has exactly one cycle (retrieve+rerank -> grade
-> rewrite -> retrieve+rerank), bounded to a single iteration because
`CorrectiveController.decide` never returns `REWRITE_AND_RETRY` once
`already_retried` is `True`.
"""

from __future__ import annotations

import time
from typing import TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agentic_rag.agent.corrective_controller import CorrectiveAction, CorrectiveController
from agentic_rag.agent.query_rewriter import QueryRewriter
from agentic_rag.agent.relevance_grader import RelevanceGrade, RelevanceGrader
from agentic_rag.domain.retrieval import RetrievalResult, RetrievalStrategy, ScoredChunk
from agentic_rag.retrieval.base import Retriever
from agentic_rag.retrieval.cross_encoder_reranker import RerankerProtocol

_CANDIDATE_POOL_SIZE = 100


class _AgenticState(TypedDict):
    """LangGraph state threaded through every node of the P5 graph."""

    query_id: str
    top_k: int
    original_query_text: str
    current_query_text: str
    candidates: list[ScoredChunk]
    last_candidates: list[ScoredChunk]
    already_retried: bool
    action: CorrectiveAction
    trace: list[str]
    final_chunks: list[ScoredChunk]


class AgenticRetriever(Retriever):
    """Composes routing, retrieval, grading, and corrective retry/abstention.

    Internally a compiled LangGraph `StateGraph`; externally just another
    `Retriever`, so `PipelineFactory`/`EvaluationRunner` need no special-casing.
    """

    def __init__(
        self,
        hybrid_retriever: Retriever,
        reranker: RerankerProtocol,
        relevance_grader: RelevanceGrader,
        query_rewriter: QueryRewriter,
        corrective_controller: CorrectiveController,
        candidate_pool_size: int = _CANDIDATE_POOL_SIZE,
    ) -> None:
        self._hybrid_retriever = hybrid_retriever
        self._reranker = reranker
        self._relevance_grader = relevance_grader
        self._query_rewriter = query_rewriter
        self._corrective_controller = corrective_controller
        self._candidate_pool_size = candidate_pool_size
        self._graph: CompiledStateGraph = self._build_graph()

    def retrieve(self, query_text: str, query_id: str, top_k: int) -> RetrievalResult:
        start_time = time.perf_counter()
        initial_state: _AgenticState = {
            "query_id": query_id,
            "top_k": top_k,
            "original_query_text": query_text,
            "current_query_text": query_text,
            "candidates": [],
            "last_candidates": [],
            "already_retried": False,
            "action": CorrectiveAction.PROCEED,
            "trace": [],
            "final_chunks": [],
        }
        final_state = self._graph.invoke(initial_state)
        return RetrievalResult(
            query_id=query_id,
            strategy=RetrievalStrategy.AGENTIC_CORRECTIVE,
            ranked_chunks=final_state["final_chunks"],
            latency_seconds=time.perf_counter() - start_time,
            trace=final_state["trace"],
        )

    def _build_graph(self) -> CompiledStateGraph:
        """Wire the four decision/action nodes into the corrective-retry graph.

        retrieve_and_rerank -> grade_and_decide -(rewrite)-> rewrite -> retrieve_and_rerank
                                                 \\-(finalize)-> finalize -> END
        """
        graph: StateGraph = StateGraph(_AgenticState)
        graph.add_node("retrieve_and_rerank", self._retrieve_and_rerank_node)
        graph.add_node("grade_and_decide", self._grade_and_decide_node)
        graph.add_node("rewrite", self._rewrite_node)
        graph.add_node("finalize", self._finalize_node)

        graph.set_entry_point("retrieve_and_rerank")
        graph.add_edge("retrieve_and_rerank", "grade_and_decide")
        graph.add_conditional_edges(
            "grade_and_decide",
            self._route_after_grade,
            {"rewrite": "rewrite", "finalize": "finalize"},
        )
        graph.add_edge("rewrite", "retrieve_and_rerank")
        graph.add_edge("finalize", END)
        return graph.compile()

    def _retrieve_and_rerank_node(self, state: _AgenticState) -> dict[str, object]:
        """Run one hybrid-retrieve-then-rerank pass over the current query text."""
        query_text = state["current_query_text"]
        base_result = self._hybrid_retriever.retrieve(
            query_text, state["query_id"], self._candidate_pool_size
        )
        reranked = self._reranker.rerank(
            query_text, base_result.ranked_chunks, self._candidate_pool_size
        )
        trace = [
            *state["trace"],
            f"Retrieved+reranked {len(reranked)} candidates for '{query_text[:60]}'",
        ]
        return {"last_candidates": reranked, "trace": trace}

    def _grade_and_decide_node(self, state: _AgenticState) -> dict[str, object]:
        """Grade the current candidate pool and decide the next action.

        On a retry, the two passes were reranked against *different* query texts
        (original vs. rewritten), and cross-encoder scores are not comparable
        across queries. So instead of mixing those stale scores, union the chunks
        from both passes and RE-RANK them once against the ORIGINAL query, which
        is what we actually want to answer. Every score then shares one query, so
        the grade and the final ranking are sound.
        """
        already_retried = state["already_retried"]
        if already_retried:
            union = self._dedupe_chunks(state["candidates"], state["last_candidates"])
            candidates = self._reranker.rerank(
                state["original_query_text"], union, self._candidate_pool_size
            )
        else:
            candidates = state["last_candidates"]

        # Per-document grading: drop candidates the grader confidently labels irrelevant,
        # so the generator/metrics never see noise the reranker ranked high but shouldn't
        # have. This is what gives P5 a mechanism to return a strictly cleaner top-k than
        # P4 instead of only matching it. Chunks we keep stay in reranker order.
        per_doc_grades = self._relevance_grader.grade_each(candidates)
        filtered = [
            c for c, g in zip(candidates, per_doc_grades, strict=True)
            if g != RelevanceGrade.CONFIDENT_IRRELEVANT
        ]
        n_dropped = len(candidates) - len(filtered)
        candidates = filtered

        grading_result = self._relevance_grader.grade(candidates)
        label = "Post-retry grade" if already_retried else "Initial grade"
        trace = [
            *state["trace"],
            f"Per-doc filter dropped {n_dropped}/{len(per_doc_grades)} confidently-irrelevant",
            f"{label}: {grading_result.grade.value} (top_score={grading_result.top_score:.3f})",
        ]

        action = self._corrective_controller.decide(grading_result.grade, already_retried)
        if action == CorrectiveAction.ABSTAIN:
            trace.append("Controller decided to ABSTAIN: returning no context")
        elif action == CorrectiveAction.PROCEED:
            trace.append("Controller decided to PROCEED with the filtered candidates")

        return {"candidates": candidates, "action": action, "trace": trace}

    @staticmethod
    def _route_after_grade(state: _AgenticState) -> str:
        """Conditional-edge path function: rewrite-and-retry, or finalize."""
        return "rewrite" if state["action"] == CorrectiveAction.REWRITE_AND_RETRY else "finalize"

    def _rewrite_node(self, state: _AgenticState) -> dict[str, object]:
        """Rephrase the original query once, ahead of the single permitted retry."""
        rewritten_query = self._query_rewriter.rewrite(state["original_query_text"])
        trace = [*state["trace"], f"Rewrote query to: '{rewritten_query}'"]
        return {"current_query_text": rewritten_query, "already_retried": True, "trace": trace}

    @staticmethod
    def _finalize_node(state: _AgenticState) -> dict[str, object]:
        """Slice the final candidate pool down to `top_k`, or return none if abstaining."""
        if state["action"] == CorrectiveAction.ABSTAIN:
            return {"final_chunks": []}
        return {"final_chunks": state["candidates"][: state["top_k"]]}

    @staticmethod
    def _dedupe_chunks(
        first: list[ScoredChunk], second: list[ScoredChunk]
    ) -> list[ScoredChunk]:
        """Union two candidate lists to unique chunks (by id); the caller re-ranks them.

        Order and the carried scores are irrelevant here because the pool is
        re-scored against a single query immediately after.
        """
        unique_by_chunk_id: dict[str, ScoredChunk] = {}
        for scored_chunk in (*first, *second):
            unique_by_chunk_id.setdefault(scored_chunk.chunk.chunk_id, scored_chunk)
        return list(unique_by_chunk_id.values())

