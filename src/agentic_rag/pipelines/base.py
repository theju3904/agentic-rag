"""Shared interface for a complete RAG pipeline (retrieval + generation)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from agentic_rag.domain.evaluation import GenerationResult
from agentic_rag.domain.retrieval import RetrievalResult


class RagPipeline(ABC):
    """One end-to-end (retrieve -> generate) strategy.

    Concrete instances expose a `strategy: RetrievalStrategy` attribute
    identifying which of the five pipelines they are.
    """

    @abstractmethod
    def answer(
        self, query_id: str, question: str, top_k: int, dataset: str | None = None
    ) -> tuple[RetrievalResult, GenerationResult]:
        """Retrieve context for `question` and generate a final answer from it."""

    @abstractmethod
    def retrieve_only(self, query_id: str, question: str, top_k: int) -> RetrievalResult:
        """Retrieve context for `question` without calling the generator.

        Used by the evaluation harness to compute retrieval metrics (which
        never need a generated answer) over a full query set cheaply, saving
        the LLM generation call for a smaller, separately-sampled subset.
        """
