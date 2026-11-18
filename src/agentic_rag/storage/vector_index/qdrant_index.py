"""Qdrant-backed `VectorIndex`: the default ANN engine used by every dense,
hybrid, and agentic pipeline run in this project.
"""

from __future__ import annotations

import logging
import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from agentic_rag.storage.vector_index.base import VectorIndex, VectorSearchHit

logger = logging.getLogger(__name__)


class QdrantVectorIndex(VectorIndex):
    """Wraps a single Qdrant collection behind the `VectorIndex` interface."""

    def __init__(
        self, client: QdrantClient, collection_name: str, embedding_dimensions: int
    ) -> None:
        """
        Args:
            client: a connected `QdrantClient`.
            collection_name: the Qdrant collection to read/write (created if missing).
            embedding_dimensions: vector size; must match the embedding model used.
        """
        self._client = client
        self._collection_name = collection_name
        self._ensure_collection(embedding_dimensions)

    def upsert(self, chunk_ids: list[str], embeddings: np.ndarray) -> None:
        if len(chunk_ids) != embeddings.shape[0]:
            raise ValueError("chunk_ids and embeddings must have the same length")
        points = [
            PointStruct(
                id=self._point_id(chunk_id),
                vector=embeddings[i].tolist(),
                payload={"chunk_id": chunk_id},
            )
            for i, chunk_id in enumerate(chunk_ids)
        ]
        self._client.upsert(collection_name=self._collection_name, points=points)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[VectorSearchHit]:
        hits = self._client.query_points(
            collection_name=self._collection_name, query=query_embedding.tolist(), limit=top_k
        ).points
        return [
            VectorSearchHit(chunk_id=self._require_chunk_id(hit), score=hit.score) for hit in hits
        ]

    def count(self) -> int:
        return self._client.count(collection_name=self._collection_name).count

    @staticmethod
    def _require_chunk_id(hit: object) -> str:
        """Extract `chunk_id` from a Qdrant hit's payload.

        Raises:
            RuntimeError: if the point has no payload, or no `chunk_id` in
                it, which would mean it wasn't written by `upsert` above.
        """
        payload = getattr(hit, "payload", None)
        if not payload or "chunk_id" not in payload:
            point_id = getattr(hit, "id", "?")
            raise RuntimeError(f"Qdrant point {point_id} is missing its chunk_id payload")
        return payload["chunk_id"]

    def _ensure_collection(self, embedding_dimensions: int) -> None:
        if self._client.collection_exists(self._collection_name):
            vectors_config = (
                self._client.get_collection(self._collection_name).config.params.vectors
            )
            # Qdrant collections created with a single unnamed vector return
            # `VectorParams`; multi-vector collections return a dict. This
            # project only ever creates the single-vector shape via
            # `_create_collection`, so anything else is a schema mismatch.
            if not isinstance(vectors_config, VectorParams):
                raise RuntimeError(
                    f"Qdrant collection '{self._collection_name}' uses a multi-vector "
                    "schema this project does not create; drop it and re-index."
                )
            existing_dim = vectors_config.size
            if existing_dim == embedding_dimensions:
                return
            logger.warning(
                "Collection '%s' has dim=%d but model requires dim=%d — "
                "dropping and recreating (re-index required)",
                self._collection_name, existing_dim, embedding_dimensions,
            )
            self._client.delete_collection(self._collection_name)
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=embedding_dimensions, distance=Distance.COSINE),
        )
        logger.info(
            "Created Qdrant collection '%s' (dim=%d)", self._collection_name, embedding_dimensions
        )

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        """Qdrant point ids must be an unsigned int or a UUID; derive a stable
        UUID5 from the chunk id and keep the real id in the payload instead.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
