"""Turns a `LoadedDataset` into persisted, chunked, embedded, and indexed data.

This is the ingestion path: `cli.py`'s `index` command is a thin wrapper
around `IndexingService.index_dataset`.
"""

from __future__ import annotations

import logging

from agentic_rag.data.base_loader import LoadedDataset
from agentic_rag.data.chunking import build_chunker_for_source
from agentic_rag.domain.document import Chunk
from agentic_rag.retrieval.embedding_model import EmbeddingModelProtocol
from agentic_rag.storage.postgres.repository import (
    ChunkRepository,
    DocumentRepository,
    QueryRepository,
)
from agentic_rag.storage.vector_index.base import VectorIndex

logger = logging.getLogger(__name__)

_EMBEDDING_BATCH_SIZE = 256


class IndexingService:
    """Chunks documents, persists them to Postgres, and populates the vector index."""

    def __init__(
        self,
        document_repository: DocumentRepository,
        chunk_repository: ChunkRepository,
        query_repository: QueryRepository,
        embedding_model: EmbeddingModelProtocol,
        vector_index: VectorIndex,
    ) -> None:
        self._document_repository = document_repository
        self._chunk_repository = chunk_repository
        self._query_repository = query_repository
        self._embedding_model = embedding_model
        self._vector_index = vector_index

    def index_dataset(self, dataset: LoadedDataset) -> int:
        """Persist documents/queries/qrels, chunk every document, embed, and index.

        Returns:
            The number of chunks produced and indexed.
        """
        logger.info(
            "Persisting %d documents, %d queries, %d qrels",
            len(dataset.documents),
            len(dataset.queries),
            len(dataset.judgments),
        )
        self._document_repository.bulk_upsert(dataset.documents)
        self._query_repository.bulk_upsert_queries(dataset.queries)
        self._query_repository.bulk_upsert_judgments(dataset.judgments)

        all_chunks = self._chunk_all_documents(dataset)
        logger.info("Produced %d chunks from %d documents", len(all_chunks), len(dataset.documents))
        self._chunk_repository.bulk_upsert(all_chunks)

        self._embed_and_index(all_chunks)
        return len(all_chunks)

    def _chunk_all_documents(self, dataset: LoadedDataset) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        for document in dataset.documents:
            chunker = build_chunker_for_source(document.source)
            all_chunks.extend(chunker.chunk(document))
        return all_chunks

    def _embed_and_index(self, chunks: list[Chunk]) -> None:
        total_chunks = len(chunks)
        for batch_start in range(0, total_chunks, _EMBEDDING_BATCH_SIZE):
            batch = chunks[batch_start : batch_start + _EMBEDDING_BATCH_SIZE]
            embeddings = self._embedding_model.encode([chunk.text for chunk in batch])
            self._vector_index.upsert([chunk.chunk_id for chunk in batch], embeddings)
            logger.info(
                "Embedded+indexed %d/%d chunks",
                min(batch_start + _EMBEDDING_BATCH_SIZE, total_chunks),
                total_chunks,
            )
