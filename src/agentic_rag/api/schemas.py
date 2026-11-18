"""Pydantic request/response models for the public API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    document_id: str
    source: str
    title: str
    text: str
    metadata: dict


class ChunkResponse(BaseModel):
    chunk_id: str
    document_id: str
    source: str
    text: str
    position: int
    metadata: dict


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The natural-language question to answer.")
    strategy: str = Field(
        ...,
        description=(
            "A top-level pipeline (sparse_bm25, dense, hybrid_rrf, hybrid_reranked, "
            "agentic_corrective) or the agentic sub-strategy name `learned_router`. The "
            "sub-strategy is accepted only when a trained artifact was loaded at startup "
            "and shares its retriever with the agentic pipeline's `learned_router` mode."
        ),
    )
    top_k: int = Field(5, ge=1, le=50)


class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    rank: int
    stage_scores: dict


class QueryResponse(BaseModel):
    query_id: str
    strategy: str
    retrieved_chunks: list[RetrievedChunkResponse]
    retrieval_latency_seconds: float
    trace: list[str]
    answer: str
    abstained: bool
    generation_latency_seconds: float


class CompareRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)


class PipelineInfoResponse(BaseModel):
    strategy: str
    description: str
