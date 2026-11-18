"""Types produced by the retrieval layer and consumed by generation/evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from agentic_rag.domain.document import Chunk


class RetrievalStrategy(StrEnum):
    """Identifiers written to `pipeline_run_records.strategy`.

    **Five top-level pipelines** are the user-facing comparison surface. The
    agentic pipeline (`AGENTIC_CORRECTIVE`) has three modes selected by
    `AGENTIC_MODE`: `learned_router` (default when a trained artifact exists),
    `heuristic_router`, and `per_doc`. `learned_router` runs BM25, dense,
    hybrid+rerank and the heuristic-router agentic branch, extracts cheap
    per-query features, and a trained XGBoost classifier picks which pool's
    results to return.

    **One agentic sub-strategy name**, `LEARNED_ROUTER` (historically "P6"),
    exists only because the training pipeline and historical ledger rows
    reference it. The retriever behind it is the same object the agentic
    pipeline's `learned_router` mode uses. It is NOT a separate pipeline, so
    the public listings (`/pipelines`, `/compare`, the demo UI, and
    `agentic-rag evaluate` by default) iterate only the top-level pipelines
    returned by `top_level_pipelines()`. The sub-strategy is still addressable
    by name through `/query`, `agentic-rag evaluate --only-strategy
    learned_router`, and `agentic-rag train-router`, which write ledger rows
    under `strategy='learned_router'` so training and reproduction stay
    deterministic.
    """

    # Top-level pipelines (P1-P5). The user-facing comparison surface.
    SPARSE_BM25 = "sparse_bm25"
    DENSE = "dense"
    HYBRID_RRF = "hybrid_rrf"
    HYBRID_RERANKED = "hybrid_reranked"
    AGENTIC_CORRECTIVE = "agentic_corrective"

    # Agentic sub-strategy name. See the class docstring.
    LEARNED_ROUTER = "learned_router"

    @classmethod
    def top_level_pipelines(cls) -> tuple[RetrievalStrategy, ...]:
        """Return the five top-level pipelines, in the project's canonical order."""
        return (
            cls.SPARSE_BM25,
            cls.DENSE,
            cls.HYBRID_RRF,
            cls.HYBRID_RERANKED,
            cls.AGENTIC_CORRECTIVE,
        )

    @property
    def is_top_level_pipeline(self) -> bool:
        """True for P1-P5, False for sub-strategy names like `LEARNED_ROUTER`."""
        return self in self.top_level_pipelines()


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A chunk with the score assigned to it by the final stage of a pipeline."""

    chunk: Chunk
    score: float
    rank: int
    stage_scores: dict = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalResult:
    """Everything a caller needs to inspect what a pipeline retrieved and why.

    Kept mutable (unlike most domain types) because the agentic pipeline
    appends human-readable trace events as it makes corrective decisions.
    """

    query_id: str
    strategy: RetrievalStrategy
    ranked_chunks: list[ScoredChunk]
    latency_seconds: float
    trace: list[str] = field(default_factory=list)

    def top_k_texts(self, k: int) -> list[str]:
        """Return the raw text of the top-`k` ranked chunks, for prompting."""
        return [scored.chunk.text for scored in self.ranked_chunks[:k]]

    def top_k_document_ids(self, k: int) -> list[str]:
        """Return the owning document id of the top-`k` ranked chunks.

        Retrieval metrics (Recall@k, nDCG@10, ...) are judged at document
        granularity in every source dataset's qrels, even though retrieval
        itself operates over chunks.
        """
        seen: dict[str, None] = {}
        for scored in self.ranked_chunks[:k]:
            seen.setdefault(scored.chunk.document_id, None)
        return list(seen.keys())
