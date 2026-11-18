"""Dense bi-encoder retriever (P2): semantic-only baseline backed by a `VectorIndex`."""

from __future__ import annotations

import time

from agentic_rag.domain.retrieval import RetrievalResult, RetrievalStrategy, ScoredChunk
from agentic_rag.retrieval.base import Retriever
from agentic_rag.retrieval.embedding_model import EmbeddingModelProtocol
from agentic_rag.storage.postgres.repository import ChunkRepository
from agentic_rag.storage.vector_index.base import VectorIndex


class DenseRetriever(Retriever):
    """Encodes the query, ANN-searches a `VectorIndex`, then hydrates full chunks."""

    def __init__(
        self,
        embedding_model: EmbeddingModelProtocol,
        vector_index: VectorIndex,
        chunk_repository: ChunkRepository,
    ) -> None:
        self._embedding_model = embedding_model
        self._vector_index = vector_index
        self._chunk_repository = chunk_repository

    def retrieve(self, query_text: str, query_id: str, top_k: int) -> RetrievalResult:
        start_time = time.perf_counter()
        query_embedding = self._embedding_model.encode_one(query_text)
        hits = self._vector_index.search(query_embedding, top_k=top_k)

        chunks_by_id = {
            chunk.chunk_id: chunk
            for chunk in self._chunk_repository.get_many([hit.chunk_id for hit in hits])
        }
        ranked_chunks = [
            ScoredChunk(chunk=chunks_by_id[hit.chunk_id], score=hit.score, rank=rank)
            for rank, hit in enumerate(hits)
            if hit.chunk_id in chunks_by_id
        ]

        return RetrievalResult(
            query_id=query_id,
            strategy=RetrievalStrategy.DENSE,
            ranked_chunks=ranked_chunks,
            latency_seconds=time.perf_counter() - start_time,
            trace=[
                f"Dense ANN search returned {len(hits)} hits, hydrated {len(ranked_chunks)} chunks"
            ],
        )
