"""SQLAlchemy ORM models for the Postgres system of record.

Postgres stores documents, chunks, relevance judgments, and the full
evaluation ledger; it is the backend for every "inspect a chunk/document" API
endpoint. Dense vectors live in Qdrant by default (see
`storage/vector_index/qdrant_index.py`); the `embedding` column below is only
populated when the alternative `PgVectorIndex` backend is selected instead.
"""

from __future__ import annotations

import datetime as dt

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIMENSIONS = 384  # sentence-transformers/all-MiniLM-L6-v2 output size


class Base(DeclarativeBase):
    """Declarative base shared by every ORM model in this project."""


class DocumentRow(Base):
    """One source document (pre-chunking) from any of the combined datasets."""

    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(String, primary_key=True)
    source: Mapped[str] = mapped_column(String, index=True, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False, default="")
    text: Mapped[str] = mapped_column(Text, nullable=False)
    doc_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(default=dt.datetime.utcnow)

    chunks: Mapped[list[ChunkRow]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class ChunkRow(Base):
    """One retrievable chunk produced by chunking a `DocumentRow`."""

    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String, primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.document_id"), index=True)
    source: Mapped[str] = mapped_column(String, index=True, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS), nullable=True
    )

    document: Mapped[DocumentRow] = relationship(back_populates="chunks")


class QueryRow(Base):
    """One benchmark question from any of the combined datasets."""

    __tablename__ = "queries"

    query_id: Mapped[str] = mapped_column(String, primary_key=True)
    source: Mapped[str] = mapped_column(String, index=True, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    query_type: Mapped[str] = mapped_column(String, nullable=False)
    gold_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    query_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)


class RelevanceJudgmentRow(Base):
    """One (query, document) relevance label -- a single qrels row."""

    __tablename__ = "relevance_judgments"
    __table_args__ = (UniqueConstraint("query_id", "document_id", name="uq_qrel_pair"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    document_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    relevance: Mapped[int] = mapped_column(Integer, nullable=False)


class EvaluationRunRow(Base):
    """One execution of the evaluation harness across all five pipelines."""

    __tablename__ = "evaluation_runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    started_at: Mapped[dt.datetime] = mapped_column(default=dt.datetime.utcnow)
    finished_at: Mapped[dt.datetime | None] = mapped_column(nullable=True)
    config_snapshot: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")


class PipelineRunRecordRow(Base):
    """One (pipeline, query) execution: retrieved chunks, answer, and metrics."""

    __tablename__ = "pipeline_run_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("evaluation_runs.run_id"), index=True)
    query_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    dataset: Mapped[str] = mapped_column(String, index=True, nullable=False)
    strategy: Mapped[str] = mapped_column(String, index=True, nullable=False)
    retrieved_chunk_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    retrieved_document_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    retrieval_latency_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    answer_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    abstained: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    generation_latency_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metrics: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
