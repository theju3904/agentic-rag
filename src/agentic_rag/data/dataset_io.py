"""Persist/reload a `LoadedDataset` as JSONL.

Keeping the golden dataset on disk means every experiment run works from an
identical, version-controllable snapshot instead of re-downloading and
re-sampling from Hugging Face (and potentially getting different rows) on
every invocation.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agentic_rag.data.base_loader import LoadedDataset
from agentic_rag.domain.document import DatasetSource, Document
from agentic_rag.domain.evaluation import RelevanceJudgment
from agentic_rag.domain.query import Query, QueryType

_DOCUMENTS_FILE = "corpus.jsonl"
_QUERIES_FILE = "queries.jsonl"
_QRELS_FILE = "qrels.jsonl"


def save_dataset_jsonl(dataset: LoadedDataset, directory: Path) -> None:
    """Write `dataset` to `directory` as three JSONL files (corpus/queries/qrels)."""
    directory.mkdir(parents=True, exist_ok=True)
    _write_jsonl(directory / _DOCUMENTS_FILE, (asdict(doc) for doc in dataset.documents))
    _write_jsonl(directory / _QUERIES_FILE, (asdict(query) for query in dataset.queries))
    _write_jsonl(directory / _QRELS_FILE, (asdict(j) for j in dataset.judgments))


def load_dataset_jsonl(directory: Path) -> LoadedDataset:
    """Reload a `LoadedDataset` previously written by `save_dataset_jsonl`.

    Raises:
        FileNotFoundError: if `directory` has no previously built golden dataset.
    """
    documents_path = directory / _DOCUMENTS_FILE
    if not documents_path.exists():
        raise FileNotFoundError(
            f"No golden dataset found at {directory}; run `agentic-rag build-dataset` first"
        )

    documents = [
        Document(
            document_id=row["document_id"],
            source=DatasetSource(row["source"]),
            title=row["title"],
            text=row["text"],
            metadata=row["metadata"],
        )
        for row in _read_jsonl(documents_path)
    ]
    queries = [
        Query(
            query_id=row["query_id"],
            source=DatasetSource(row["source"]),
            text=row["text"],
            query_type=QueryType(row["query_type"]),
            gold_answer=row["gold_answer"],
            metadata=row["metadata"],
        )
        for row in _read_jsonl(directory / _QUERIES_FILE)
    ]
    judgments = [
        RelevanceJudgment(
            query_id=row["query_id"],
            document_id=row["document_id"],
            relevance=row["relevance"],
        )
        for row in _read_jsonl(directory / _QRELS_FILE)
    ]
    return LoadedDataset(documents=documents, queries=queries, judgments=judgments)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)
