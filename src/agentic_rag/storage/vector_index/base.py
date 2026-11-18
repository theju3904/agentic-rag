"""Abstract interface for dense-vector nearest-neighbor search backends.

Two implementations exist (`QdrantVectorIndex`, `PgVectorIndex`) so the
project can demonstrate, and let a reader compare, both a dedicated vector
database and a "just use Postgres" approach. See README, "Storage design",
for why Qdrant is the default used in the experiments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class VectorSearchHit:
    """One nearest-neighbor result: a chunk id and its similarity score."""

    chunk_id: str
    score: float


class VectorIndex(ABC):
    """A collection of (chunk_id -> embedding) pairs supporting ANN search."""

    @abstractmethod
    def upsert(self, chunk_ids: list[str], embeddings: np.ndarray) -> None:
        """Insert or replace the embedding for each id in `chunk_ids`.

        Args:
            chunk_ids: ids aligned row-for-row with `embeddings`.
            embeddings: a 2D array of shape (len(chunk_ids), embedding_dim).
        """

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int) -> list[VectorSearchHit]:
        """Return the `top_k` chunk ids whose embeddings are closest to `query_embedding`."""

    @abstractmethod
    def count(self) -> int:
        """Return how many vectors are currently stored in the index."""
