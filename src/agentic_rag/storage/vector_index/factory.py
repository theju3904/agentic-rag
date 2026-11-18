"""Constructs the configured `VectorIndex` backend from `Settings`.

Shared by `pipelines/factory.py` (query-time) and `cli.py`'s `index` command
(ingestion-time) so there is exactly one place that decides Qdrant vs. pgvector.
"""

from __future__ import annotations

from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from agentic_rag.config import Settings
from agentic_rag.retrieval.embedding_model import EmbeddingModelProtocol
from agentic_rag.storage.vector_index.base import VectorIndex
from agentic_rag.storage.vector_index.pgvector_index import PgVectorIndex
from agentic_rag.storage.vector_index.qdrant_index import QdrantVectorIndex


def build_vector_index(
    settings: Settings, embedding_model: EmbeddingModelProtocol, session: Session | None = None
) -> VectorIndex:
    """Return the `VectorIndex` selected by `settings.vector_backend` ("qdrant"|"pgvector").

    Raises:
        ValueError: if "pgvector" is selected without an active `session`.
    """
    if settings.vector_backend == "pgvector":
        if session is None:
            raise ValueError("vector_backend='pgvector' requires an active SQLAlchemy session")
        return PgVectorIndex(session)
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    return QdrantVectorIndex(client, settings.qdrant_collection_name, embedding_model.dimensions)
