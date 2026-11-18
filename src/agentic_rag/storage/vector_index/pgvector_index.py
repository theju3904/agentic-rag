"""Postgres+pgvector-backed `VectorIndex`: a dependency-light alternative to
Qdrant for small deployments that would rather not run a second database.
Kept at feature parity with `QdrantVectorIndex` so pipelines can swap between
the two via configuration alone (see `config.Settings.vector_backend`).
"""

from __future__ import annotations

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from agentic_rag.storage.postgres.models import ChunkRow
from agentic_rag.storage.vector_index.base import VectorIndex, VectorSearchHit


class PgVectorIndex(VectorIndex):
    """Stores embeddings in `chunks.embedding` and searches with pgvector's `<=>` operator."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert(self, chunk_ids: list[str], embeddings: np.ndarray) -> None:
        if len(chunk_ids) != embeddings.shape[0]:
            raise ValueError("chunk_ids and embeddings must have the same length")
        for chunk_id, embedding in zip(chunk_ids, embeddings, strict=True):
            self._session.query(ChunkRow).filter(ChunkRow.chunk_id == chunk_id).update(
                {"embedding": embedding.tolist()}
            )
        self._session.commit()

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[VectorSearchHit]:
        query_vector_literal = str(query_embedding.tolist())
        rows = self._session.execute(
            text(
                "SELECT chunk_id, 1 - (embedding <=> CAST(:query_vector AS vector)) AS score "
                "FROM chunks WHERE embedding IS NOT NULL "
                "ORDER BY embedding <=> CAST(:query_vector AS vector) LIMIT :top_k"
            ),
            {"query_vector": query_vector_literal, "top_k": top_k},
        )
        return [VectorSearchHit(chunk_id=row.chunk_id, score=row.score) for row in rows]

    def count(self) -> int:
        return self._session.query(ChunkRow).filter(ChunkRow.embedding.isnot(None)).count()
