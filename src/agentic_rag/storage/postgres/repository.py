"""Repository layer: the only place that translates between domain dataclasses
and SQLAlchemy ORM rows. Retrieval/generation/evaluation code never imports
SQLAlchemy directly; it depends on these repositories instead.
"""

from __future__ import annotations

import datetime as dt
import logging

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from agentic_rag.domain.document import Chunk, DatasetSource, Document
from agentic_rag.domain.evaluation import GenerationResult, PipelineRunRecord, RelevanceJudgment
from agentic_rag.domain.query import Query, QueryType
from agentic_rag.storage.postgres.models import (
    ChunkRow,
    DocumentRow,
    EvaluationRunRow,
    PipelineRunRecordRow,
    QueryRow,
    RelevanceJudgmentRow,
)

logger = logging.getLogger(__name__)

_UPSERT_BATCH_SIZE = 500


class DocumentRepository:
    """Persists and retrieves `Document` entities."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def bulk_upsert(self, documents: list[Document]) -> None:
        """Insert or update `documents`, batched to keep single statements small."""
        for batch in _batched(documents, _UPSERT_BATCH_SIZE):
            rows = [
                {
                    "document_id": doc.document_id,
                    "source": doc.source.value,
                    "title": doc.title,
                    "text": doc.text,
                    "doc_metadata": doc.metadata,
                }
                for doc in batch
            ]
            statement = pg_insert(DocumentRow).values(rows)
            statement = statement.on_conflict_do_update(
                index_elements=[DocumentRow.document_id],
                set_={
                    "title": statement.excluded.title,
                    "text": statement.excluded.text,
                    "doc_metadata": statement.excluded.doc_metadata,
                },
            )
            self._session.execute(statement)
        self._session.commit()

    def get(self, document_id: str) -> Document | None:
        """Return the document with `document_id`, or `None` if it doesn't exist."""
        row = self._session.get(DocumentRow, document_id)
        return self._to_domain(row) if row else None

    def count(self) -> int:
        return self._session.query(DocumentRow).count()

    @staticmethod
    def _to_domain(row: DocumentRow) -> Document:
        return Document(
            document_id=row.document_id,
            source=DatasetSource(row.source),
            title=row.title,
            text=row.text,
            metadata=row.doc_metadata,
        )


class ChunkRepository:
    """Persists and retrieves `Chunk` entities."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def bulk_upsert(self, chunks: list[Chunk]) -> None:
        """Insert or update `chunks`, batched to keep single statements small."""
        for batch in _batched(chunks, _UPSERT_BATCH_SIZE):
            rows = [
                {
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "source": c.source.value,
                    "text": c.text,
                    "position": c.position,
                    "chunk_metadata": c.metadata,
                }
                for c in batch
            ]
            statement = pg_insert(ChunkRow).values(rows)
            statement = statement.on_conflict_do_update(
                index_elements=[ChunkRow.chunk_id],
                set_={
                    "text": statement.excluded.text,
                    "position": statement.excluded.position,
                    "chunk_metadata": statement.excluded.chunk_metadata,
                },
            )
            self._session.execute(statement)
        self._session.commit()

    def get(self, chunk_id: str) -> Chunk | None:
        """Return the chunk with `chunk_id`, or `None` if it doesn't exist."""
        row = self._session.get(ChunkRow, chunk_id)
        return self._to_domain(row) if row else None

    def get_many(self, chunk_ids: list[str]) -> list[Chunk]:
        """Fetch chunks by id, preserving the order of `chunk_ids`.

        Any id with no matching row is skipped and logged, rather than
        raising, so one stale reference can't fail an entire request.
        """
        rows = (
            self._session.execute(select(ChunkRow).where(ChunkRow.chunk_id.in_(chunk_ids)))
            .scalars()
            .all()
        )
        chunk_by_id = {row.chunk_id: self._to_domain(row) for row in rows}
        missing = [cid for cid in chunk_ids if cid not in chunk_by_id]
        if missing:
            logger.warning("ChunkRepository.get_many: %d chunk ids not found", len(missing))
        return [chunk_by_id[cid] for cid in chunk_ids if cid in chunk_by_id]

    def list_by_document(self, document_id: str) -> list[Chunk]:
        """Return every chunk belonging to `document_id`, in reading order."""
        rows = (
            self._session.execute(
                select(ChunkRow)
                .where(ChunkRow.document_id == document_id)
                .order_by(ChunkRow.position)
            )
            .scalars()
            .all()
        )
        return [self._to_domain(row) for row in rows]

    def list_all(self) -> list[Chunk]:
        """Return every chunk. Used to build the in-memory BM25 index at startup."""
        rows = self._session.execute(select(ChunkRow)).scalars().all()
        return [self._to_domain(row) for row in rows]

    def count(self) -> int:
        return self._session.query(ChunkRow).count()

    @staticmethod
    def _to_domain(row: ChunkRow) -> Chunk:
        return Chunk(
            chunk_id=row.chunk_id,
            document_id=row.document_id,
            source=DatasetSource(row.source),
            text=row.text,
            position=row.position,
            metadata=row.chunk_metadata,
        )


class QueryRepository:
    """Persists `Query` entities and their `RelevanceJudgment`s (qrels)."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def bulk_upsert_queries(self, queries: list[Query]) -> None:
        for batch in _batched(queries, _UPSERT_BATCH_SIZE):
            rows = [
                {
                    "query_id": q.query_id,
                    "source": q.source.value,
                    "text": q.text,
                    "query_type": q.query_type.value,
                    "gold_answer": q.gold_answer,
                    "query_metadata": q.metadata,
                }
                for q in batch
            ]
            statement = pg_insert(QueryRow).values(rows)
            statement = statement.on_conflict_do_update(
                index_elements=[QueryRow.query_id],
                set_={
                    "text": statement.excluded.text,
                    "query_type": statement.excluded.query_type,
                    "gold_answer": statement.excluded.gold_answer,
                    "query_metadata": statement.excluded.query_metadata,
                },
            )
            self._session.execute(statement)
        self._session.commit()

    def bulk_upsert_judgments(self, judgments: list[RelevanceJudgment]) -> None:
        """Insert or update qrels, deduplicated by (query_id, document_id).

        Source datasets occasionally cite the same document more than once
        for one query (e.g. MultiHop-RAG's evidence list can reference one
        article for multiple facts); Postgres's `ON CONFLICT DO UPDATE`
        cannot affect the same row twice within a single statement, so
        duplicates within a batch must be collapsed before inserting.
        """
        deduplicated = list({(j.query_id, j.document_id): j for j in judgments}.values())
        for batch in _batched(deduplicated, _UPSERT_BATCH_SIZE):
            rows = [
                {"query_id": j.query_id, "document_id": j.document_id, "relevance": j.relevance}
                for j in batch
            ]
            statement = pg_insert(RelevanceJudgmentRow).values(rows)
            statement = statement.on_conflict_do_update(
                index_elements=["query_id", "document_id"],
                set_={"relevance": statement.excluded.relevance},
            )
            self._session.execute(statement)
        self._session.commit()

    def list_all(self) -> list[Query]:
        rows = self._session.execute(select(QueryRow)).scalars().all()
        return [self._to_domain(row) for row in rows]

    def get(self, query_id: str) -> Query | None:
        row = self._session.get(QueryRow, query_id)
        return self._to_domain(row) if row else None

    def judgments_by_query(self) -> dict[str, dict[str, int]]:
        """Return qrels as `{query_id: {document_id: relevance}}`, the shape
        every retrieval metric in `evaluation/retrieval_metrics.py` expects.
        """
        rows = self._session.execute(select(RelevanceJudgmentRow)).scalars().all()
        grouped: dict[str, dict[str, int]] = {}
        for row in rows:
            grouped.setdefault(row.query_id, {})[row.document_id] = row.relevance
        return grouped

    @staticmethod
    def _to_domain(row: QueryRow) -> Query:
        return Query(
            query_id=row.query_id,
            source=DatasetSource(row.source),
            text=row.text,
            query_type=QueryType(row.query_type),
            gold_answer=row.gold_answer,
            metadata=row.query_metadata,
        )


class EvaluationRepository:
    """Persists the evaluation ledger: one row per (pipeline, query) execution."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def start_run(self, run_id: str, config_snapshot: dict) -> None:
        """Create a new `evaluation_runs` row. Raises if `run_id` already exists."""
        self._session.add(EvaluationRunRow(run_id=run_id, config_snapshot=config_snapshot))
        self._session.commit()

    def finish_run(self, run_id: str) -> None:
        row = self._session.get(EvaluationRunRow, run_id)
        if row is None:
            raise ValueError(f"Unknown evaluation run_id '{run_id}'")
        row.finished_at = dt.datetime.utcnow()
        self._session.commit()

    def save_record(
        self, run_id: str, record: PipelineRunRecord, generation: GenerationResult | None
    ) -> None:
        """Persist one pipeline/query execution as a single ledger row."""
        row = PipelineRunRecordRow(
            run_id=run_id,
            query_id=record.query_id,
            dataset=record.dataset,
            strategy=record.strategy.value,
            retrieved_chunk_ids=[sc.chunk.chunk_id for sc in record.retrieval.ranked_chunks],
            retrieved_document_ids=record.retrieval.top_k_document_ids(
                len(record.retrieval.ranked_chunks)
            ),
            retrieval_latency_seconds=record.retrieval.latency_seconds,
            answer_text=generation.answer_text if generation else None,
            abstained=generation.abstained if generation else None,
            generation_latency_seconds=generation.latency_seconds if generation else None,
            prompt_tokens=generation.prompt_tokens if generation else None,
            completion_tokens=generation.completion_tokens if generation else None,
            metrics=record.metrics,
        )
        self._session.add(row)
        self._session.commit()

    def fetch_run_records(self, run_id: str) -> list[PipelineRunRecordRow]:
        return list(
            self._session.execute(
                select(PipelineRunRecordRow).where(PipelineRunRecordRow.run_id == run_id)
            ).scalars()
        )

    def run_exists(self, run_id: str) -> bool:
        """True if an `evaluation_runs` row already exists for `run_id`."""
        return self._session.get(EvaluationRunRow, run_id) is not None

    def completed_keys(self, run_id: str) -> set[tuple[str, str]]:
        """Return `{(strategy, query_id), ...}` already recorded under `run_id`.

        Used by the CLI's `--append-to` resume path: the runner skips any
        (pipeline, query) pair in this set so a long eval can be split across
        multiple background runs without redoing work. Scales fine via an index
        on `run_id` plus the per-row primary key; a run with ~20k records
        materialises in well under a second.
        """
        rows = self._session.execute(
            select(PipelineRunRecordRow.strategy, PipelineRunRecordRow.query_id)
            .where(PipelineRunRecordRow.run_id == run_id)
        ).all()
        return {(strategy, query_id) for strategy, query_id in rows}


def _batched(items: list, batch_size: int):
    """Yield successive `batch_size`-sized slices of `items`."""
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
