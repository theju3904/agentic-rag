"""Loader for BEIR-format datasets served from the `BeIR/*` Hugging Face mirror.

Used for SciFact, NFCorpus, and FiQA-2018 (research doc, Tier A) but works for
any dataset published in the same `BeIR/<name>` / `BeIR/<name>-qrels` layout.
We deliberately load via `datasets` rather than the official `beir` pip
package to avoid its heavier, occasionally version-fragile dependency tree.
"""

from __future__ import annotations

import logging

from datasets import get_dataset_split_names, load_dataset

from agentic_rag.data.base_loader import DatasetLoader, DatasetLoadError, LoadedDataset
from agentic_rag.domain.document import DatasetSource, Document
from agentic_rag.domain.evaluation import RelevanceJudgment
from agentic_rag.domain.query import Query, QueryType

logger = logging.getLogger(__name__)

# BEIR ships train/dev/test qrels per dataset but not every dataset has all
# three; prefer the canonical zero-shot "test" split and fall back in order.
_PREFERRED_QRELS_SPLITS = ("test", "dev", "validation", "train")


class BeirDatasetLoader(DatasetLoader):
    """Loads one `BeIR/<hf_name>` dataset and namespaces every id by `source`."""

    def __init__(self, hf_name: str, source: DatasetSource) -> None:
        """
        Args:
            hf_name: dataset slug under the `BeIR` HF org, e.g. ``"scifact"``.
            source: the `DatasetSource` this loader's output is tagged with.
        """
        self._hf_name = hf_name
        self._source = source

    def load(self) -> LoadedDataset:
        try:
            corpus_rows = load_dataset("BeIR/" + self._hf_name, "corpus", split="corpus")
            query_rows = load_dataset("BeIR/" + self._hf_name, "queries", split="queries")
            qrels_rows = load_dataset(
                f"BeIR/{self._hf_name}-qrels", split=self._pick_qrels_split()
            )
        except DatasetLoadError:
            raise
        except Exception as exc:  # network failure, missing config, schema drift, ...
            raise DatasetLoadError(
                f"Failed to load BEIR dataset '{self._hf_name}' from Hugging Face"
            ) from exc

        judgments = [
            RelevanceJudgment(
                query_id=self._namespaced(str(row["query-id"])),
                document_id=self._namespaced(str(row["corpus-id"])),
                relevance=int(row["score"]),
            )
            for row in qrels_rows
        ]
        # The "queries" config bundles every split's questions together; only
        # keep the ones our chosen qrels split actually has judgments for.
        evaluated_query_ids = {judgment.query_id for judgment in judgments}

        documents = [
            Document(
                document_id=self._namespaced(str(row["_id"])),
                source=self._source,
                title=row.get("title") or "",
                text=row["text"],
                metadata={},
            )
            for row in corpus_rows
        ]
        queries = [
            Query(
                query_id=self._namespaced(str(row["_id"])),
                source=self._source,
                text=row["text"],
                query_type=QueryType.SIMPLE,
            )
            for row in query_rows
            if self._namespaced(str(row["_id"])) in evaluated_query_ids
        ]

        if not documents or not queries or not judgments:
            raise DatasetLoadError(
                f"BEIR dataset '{self._hf_name}' loaded but is empty: "
                f"{len(documents)} docs, {len(queries)} queries, {len(judgments)} qrels"
            )

        logger.info(
            "Loaded BEIR dataset '%s': %d docs, %d queries, %d qrels",
            self._hf_name,
            len(documents),
            len(queries),
            len(judgments),
        )
        return LoadedDataset(documents=documents, queries=queries, judgments=judgments)

    def _pick_qrels_split(self) -> str:
        """Prefer the canonical "test" qrels split, falling back defensively."""
        try:
            available = set(get_dataset_split_names(f"BeIR/{self._hf_name}-qrels"))
        except Exception as exc:
            raise DatasetLoadError(
                f"Could not list qrels splits for BEIR dataset '{self._hf_name}'"
            ) from exc
        for candidate in _PREFERRED_QRELS_SPLITS:
            if candidate in available:
                return candidate
        raise DatasetLoadError(
            f"BEIR dataset '{self._hf_name}-qrels' has no recognized split "
            f"(found {sorted(available)})"
        )

    def _namespaced(self, raw_id: str) -> str:
        """Prefix a raw BEIR id with the dataset source to keep ids globally unique."""
        return f"{self._source.value}::{raw_id}"
