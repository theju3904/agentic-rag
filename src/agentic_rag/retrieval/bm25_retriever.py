"""BM25 sparse retriever (P1): the lexical baseline every pipeline is compared against."""

from __future__ import annotations

import re
import time

from rank_bm25 import BM25Okapi

from agentic_rag.domain.document import Chunk
from agentic_rag.domain.retrieval import RetrievalResult, RetrievalStrategy, ScoredChunk
from agentic_rag.retrieval.base import Retriever

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-only tokenization shared by indexing and querying."""
    return _TOKEN_PATTERN.findall(text.lower())


class Bm25Retriever(Retriever):
    """Wraps `rank_bm25.BM25Okapi` over a fixed, in-memory chunk collection."""

    def __init__(self, chunks: list[Chunk]) -> None:
        """
        Args:
            chunks: every chunk in the corpus this retriever should search over.

        Raises:
            ValueError: if `chunks` is empty, since there would be nothing to index.
        """
        if not chunks:
            raise ValueError("Bm25Retriever requires at least one chunk to index")
        self._chunks = chunks
        self._index = BM25Okapi([tokenize(chunk.text) for chunk in chunks])

    def retrieve(self, query_text: str, query_id: str, top_k: int) -> RetrievalResult:
        start_time = time.perf_counter()
        scores = self._index.get_scores(tokenize(query_text))
        ranked_indices = scores.argsort()[::-1][:top_k]
        ranked_chunks = [
            ScoredChunk(chunk=self._chunks[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(ranked_indices)
        ]
        return RetrievalResult(
            query_id=query_id,
            strategy=RetrievalStrategy.SPARSE_BM25,
            ranked_chunks=ranked_chunks,
            latency_seconds=time.perf_counter() - start_time,
            trace=[f"BM25 scored {len(self._chunks)} chunks, returned top {len(ranked_chunks)}"],
        )
