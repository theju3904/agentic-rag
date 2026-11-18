"""Lists the five top-level pipelines currently loaded in `AppContext`."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from agentic_rag.api.context import AppContext
from agentic_rag.api.dependencies import get_app_context
from agentic_rag.api.schemas import PipelineInfoResponse
from agentic_rag.domain.retrieval import RetrievalStrategy

router = APIRouter(prefix="/pipelines", tags=["pipelines"])

_PIPELINE_DESCRIPTIONS = {
    "sparse_bm25": "P1: BM25 lexical search only.",
    "dense": "P2: bi-encoder embeddings + ANN search only.",
    "hybrid_rrf": "P3: BM25 + dense fused with reciprocal rank fusion.",
    "hybrid_reranked": "P4: P3 candidates re-scored by a cross-encoder.",
    "agentic_corrective": (
        "P5: agentic, per-query routing; mode set by AGENTIC_MODE "
        "(learned_router by default, else heuristic_router / per_doc)."
    ),
}


@router.get("", response_model=list[PipelineInfoResponse])
def list_pipelines(
    context: AppContext = Depends(get_app_context),
) -> list[PipelineInfoResponse]:
    """Return the five top-level pipelines currently loaded.

    The agentic sub-strategy `learned_router` is NOT listed here: it exists
    only as an addressable name for training (`agentic-rag train-router`) and
    historical ledger queries, not as a user-facing pipeline. Reads the live
    `AppContext.pipelines` so the list reflects what the factory actually
    built.
    """
    return [
        PipelineInfoResponse(
            strategy=strategy.value,
            description=_PIPELINE_DESCRIPTIONS[strategy.value],
        )
        for strategy in RetrievalStrategy.top_level_pipelines()
        if strategy in context.pipelines
    ]
