"""Types for recording evaluation outcomes, independent of how metrics are computed."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_rag.domain.retrieval import RetrievalResult, RetrievalStrategy


@dataclass(frozen=True, slots=True)
class RelevanceJudgment:
    """One (query, document) relevance label, i.e. one row of a qrels file."""

    query_id: str
    document_id: str
    relevance: int


@dataclass(slots=True)
class GenerationResult:
    """The final answer produced by a pipeline for one query, plus bookkeeping."""

    query_id: str
    strategy: RetrievalStrategy
    answer_text: str
    abstained: bool
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float


@dataclass(slots=True)
class PipelineRunRecord:
    """One row of the evaluation ledger: a single (pipeline, query) execution."""

    query_id: str
    dataset: str
    strategy: RetrievalStrategy
    retrieval: RetrievalResult
    generation: GenerationResult | None
    metrics: dict = field(default_factory=dict)
