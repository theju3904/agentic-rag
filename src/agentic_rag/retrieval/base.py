"""Shared interface every retrieval strategy (P1-P5) implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

from agentic_rag.domain.retrieval import RetrievalResult


class Retriever(ABC):
    """Ranks corpus chunks against a query. One instance per retrieval strategy."""

    @abstractmethod
    def retrieve(self, query_text: str, query_id: str, top_k: int) -> RetrievalResult:
        """Return the `top_k` best-matching chunks for `query_text`, ranked best-first."""
