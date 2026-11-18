"""Run a single question through one chosen pipeline, or through all five at once."""

from __future__ import annotations

import hashlib

from fastapi import APIRouter, Depends, HTTPException

from agentic_rag.api.context import AppContext
from agentic_rag.api.dependencies import get_app_context
from agentic_rag.api.schemas import (
    CompareRequest,
    QueryRequest,
    QueryResponse,
    RetrievedChunkResponse,
)
from agentic_rag.domain.evaluation import GenerationResult
from agentic_rag.domain.retrieval import RetrievalResult, RetrievalStrategy
from agentic_rag.pipelines.base import RagPipeline

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def run_query(
    request: QueryRequest, context: AppContext = Depends(get_app_context)
) -> QueryResponse:
    """Retrieve context and generate an answer for `request.question` with one pipeline."""
    pipeline = _resolve_pipeline(context, request.strategy)
    retrieval_result, generation_result = pipeline.answer(
        _stable_query_id(request.question), request.question, request.top_k
    )
    return _to_query_response(retrieval_result, generation_result)


@router.post("/compare", response_model=dict[str, QueryResponse])
def compare_pipelines(
    request: CompareRequest, context: AppContext = Depends(get_app_context)
) -> dict[str, QueryResponse]:
    """Run `request.question` through every top-level pipeline (P1-P5).

    The agentic sub-strategy `LEARNED_ROUTER` is NOT included: its retriever
    is the same object the agentic pipeline's `learned_router` mode uses, so
    running it would do identical work twice. Use `/query` with
    `strategy=learned_router` to address the sub-strategy directly. One
    pipeline failure currently aborts the whole response - per-pipeline error
    isolation is a planned follow-up.
    """
    query_id = _stable_query_id(request.question)
    return {
        strategy.value: _to_query_response(
            *context.pipelines[strategy].answer(query_id, request.question, request.top_k)
        )
        for strategy in RetrievalStrategy.top_level_pipelines()
        if strategy in context.pipelines
    }


def _resolve_pipeline(context: AppContext, strategy_name: str) -> RagPipeline:
    try:
        strategy = RetrievalStrategy(strategy_name)
    except ValueError as exc:
        valid_names = [strategy.value for strategy in RetrievalStrategy]
        raise HTTPException(
            status_code=422,
            detail=f"Unknown strategy '{strategy_name}'; must be one of {valid_names}",
        ) from exc

    pipeline = context.pipelines.get(strategy)
    if pipeline is None:
        # Known strategy, but not registered in this process (e.g. `learned_router`
        # without a trained artifact on disk). Return 503 so callers can tell this
        # apart from an unknown-strategy 422 or an actual server bug 500.
        raise HTTPException(
            status_code=503,
            detail=(
                f"Strategy '{strategy_name}' is known but no pipeline is loaded in this "
                "process. See GET /pipelines for what is actually available."
            ),
        )
    return pipeline


def _to_query_response(
    retrieval_result: RetrievalResult, generation_result: GenerationResult
) -> QueryResponse:
    return QueryResponse(
        query_id=retrieval_result.query_id,
        strategy=retrieval_result.strategy.value,
        retrieved_chunks=[
            RetrievedChunkResponse(
                chunk_id=scored_chunk.chunk.chunk_id,
                document_id=scored_chunk.chunk.document_id,
                text=scored_chunk.chunk.text,
                score=scored_chunk.score,
                rank=scored_chunk.rank,
                stage_scores=scored_chunk.stage_scores,
            )
            for scored_chunk in retrieval_result.ranked_chunks
        ],
        retrieval_latency_seconds=retrieval_result.latency_seconds,
        trace=retrieval_result.trace,
        answer=generation_result.answer_text,
        abstained=generation_result.abstained,
        generation_latency_seconds=generation_result.latency_seconds,
    )


def _stable_query_id(question: str) -> str:
    """A deterministic id for ad-hoc API queries (not tied to Python's randomized hash)."""
    return f"api-{hashlib.sha1(question.encode('utf-8')).hexdigest()[:12]}"
