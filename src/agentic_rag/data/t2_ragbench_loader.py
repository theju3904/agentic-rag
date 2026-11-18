"""Loader for T2-RAGBench (Strich et al., EACL 2026): finance text+table RAG.

Uses the `G4KMU/t2-ragbench-splits` mirror (~45 MB) rather than the primary
`G4KMU/t2-ragbench` repo (~3 GB) because the latter embeds the original PDF
files, which this project never needs for text retrieval.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from datasets import concatenate_datasets, get_dataset_split_names, load_dataset

from agentic_rag.data.base_loader import DatasetLoader, DatasetLoadError, LoadedDataset
from agentic_rag.domain.document import DatasetSource, Document
from agentic_rag.domain.evaluation import RelevanceJudgment
from agentic_rag.domain.query import Query, QueryType

logger = logging.getLogger(__name__)

_HF_REPO = "G4KMU/t2-ragbench-splits"
_SUBSET_CONFIGS = ("FinQA", "ConvFinQA", "TAT-DQA")


class T2RagBenchLoader(DatasetLoader):
    """Loads and pools all three T2-RAGBench subsets into one corpus/query set."""

    def load(self) -> LoadedDataset:
        rows = list(self._iter_all_rows())
        if not rows:
            raise DatasetLoadError(
                f"T2-RAGBench loaded 0 rows from '{_HF_REPO}' across configs {_SUBSET_CONFIGS}"
            )

        documents: dict[str, Document] = {}
        queries: list[Query] = []
        judgments: list[RelevanceJudgment] = []

        for subset_name, row in rows:
            try:
                context_id = str(row["context_id"])
                question = row["question"]
                query_row_id = str(row["id"])
                context_text = row["context"]
            except KeyError as exc:
                raise DatasetLoadError(
                    f"T2-RAGBench subset '{subset_name}' is missing expected column {exc}; "
                    f"available columns were {sorted(row.keys())}"
                ) from exc

            document_id = self._namespaced(f"{subset_name}::{context_id}")
            if document_id not in documents:
                documents[document_id] = Document(
                    document_id=document_id,
                    source=DatasetSource.T2_RAGBENCH,
                    title=row.get("file_name") or row.get("company_name") or "",
                    text=context_text,
                    metadata={"subset": subset_name, "company_name": row.get("company_name", "")},
                )

            query_id = self._namespaced(f"{subset_name}::{query_row_id}")
            queries.append(
                Query(
                    query_id=query_id,
                    source=DatasetSource.T2_RAGBENCH,
                    text=question,
                    query_type=QueryType.SIMPLE,
                    gold_answer=str(row.get("program_answer", row.get("original_answer", ""))),
                    metadata={"subset": subset_name, "original_split": row.get("split", "")},
                )
            )
            judgments.append(
                RelevanceJudgment(query_id=query_id, document_id=document_id, relevance=1)
            )

        logger.info(
            "Loaded T2-RAGBench: %d docs, %d queries, %d qrels",
            len(documents),
            len(queries),
            len(judgments),
        )
        return LoadedDataset(
            documents=list(documents.values()), queries=queries, judgments=judgments
        )

    def _iter_all_rows(self) -> Iterator[tuple[str, Any]]:
        """Yield (subset_name, row) for every split of every T2-RAGBench subset.

        Each subset is tried independently so a schema change in one subset
        doesn't prevent the others from loading.
        """
        for subset_name in _SUBSET_CONFIGS:
            try:
                split_names = get_dataset_split_names(_HF_REPO, subset_name)
                subset_splits = [
                    load_dataset(_HF_REPO, subset_name, split=split_name)
                    for split_name in split_names
                ]
                combined = concatenate_datasets(subset_splits)
            except Exception:
                logger.exception("Skipping T2-RAGBench subset '%s'", subset_name)
                continue
            yield from ((subset_name, row) for row in combined)

    @staticmethod
    def _namespaced(raw_id: str) -> str:
        return f"{DatasetSource.T2_RAGBENCH.value}::{raw_id}"
