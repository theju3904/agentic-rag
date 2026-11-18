"""Core document and chunk entities produced by the data ingestion layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class DatasetSource(StrEnum):
    """Identifies which source dataset a document/query/chunk originated from."""

    SCIFACT = "scifact"
    NFCORPUS = "nfcorpus"
    FIQA = "fiqa"
    MULTIHOP_RAG = "multihop_rag"
    T2_RAGBENCH = "t2_ragbench"


@dataclass(frozen=True, slots=True)
class Document:
    """A single source document before chunking.

    `document_id` is namespaced per dataset (e.g. ``"scifact::doc4983"``) so
    identifiers never collide once every dataset is combined into the golden
    corpus.
    """

    document_id: str
    source: DatasetSource
    title: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable unit of text produced by chunking a `Document`."""

    chunk_id: str
    document_id: str
    source: DatasetSource
    text: str
    position: int
    metadata: dict = field(default_factory=dict)
