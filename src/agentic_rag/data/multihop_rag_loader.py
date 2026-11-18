"""Loader for the MultiHop-RAG dataset (Tang & Yang, COLM 2024).

Source: https://huggingface.co/datasets/yixuantt/MultiHopRAG
2,556 queries whose evidence is spread across 2-4 news articles, each query
labelled with one of four types: comparison, inference, temporal, or null
(the last being deliberately unanswerable from the corpus).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from datasets import load_dataset

from agentic_rag.data.base_loader import DatasetLoader, DatasetLoadError, LoadedDataset
from agentic_rag.domain.document import DatasetSource, Document
from agentic_rag.domain.evaluation import RelevanceJudgment
from agentic_rag.domain.query import Query, QueryType

logger = logging.getLogger(__name__)

_HF_REPO = "yixuantt/MultiHopRAG"
_QUERY_TYPE_BY_RAW_LABEL = {
    "comparison_query": QueryType.COMPARISON,
    "inference_query": QueryType.INFERENCE,
    "temporal_query": QueryType.TEMPORAL,
    "null_query": QueryType.NULL,
}


class MultiHopRagLoader(DatasetLoader):
    """Loads MultiHop-RAG's news corpus and its multi-hop question set."""

    def load(self) -> LoadedDataset:
        try:
            query_rows = load_dataset(_HF_REPO, "MultiHopRAG", split="train")
            corpus_rows = load_dataset(_HF_REPO, "corpus", split="train")
        except Exception as exc:
            raise DatasetLoadError(
                f"Failed to load MultiHop-RAG from Hugging Face repo '{_HF_REPO}'"
            ) from exc

        documents, document_id_by_lookup_key = self._build_documents(corpus_rows)
        queries: list[Query] = []
        judgments: list[RelevanceJudgment] = []

        for row in query_rows:
            query_id = self._namespaced(self._stable_id(row.get("query", "")))
            raw_type = str(row.get("question_type", "")).strip().lower()
            query_type = _QUERY_TYPE_BY_RAW_LABEL.get(raw_type, QueryType.SIMPLE)
            if raw_type and raw_type not in _QUERY_TYPE_BY_RAW_LABEL:
                logger.warning("Unrecognized MultiHop-RAG question_type '%s'", raw_type)

            queries.append(
                Query(
                    query_id=query_id,
                    source=DatasetSource.MULTIHOP_RAG,
                    text=row["query"],
                    query_type=query_type,
                    gold_answer=row.get("answer"),
                    metadata={"raw_question_type": raw_type},
                )
            )
            for evidence in row.get("evidence_list", None) or []:
                lookup_key = self._lookup_key(evidence.get("url"), evidence.get("title"))
                document_id = document_id_by_lookup_key.get(lookup_key)
                if document_id is None:
                    continue  # evidence referenced an article outside the shipped corpus
                judgments.append(
                    RelevanceJudgment(query_id=query_id, document_id=document_id, relevance=1)
                )

        if not documents or not queries:
            raise DatasetLoadError(
                f"MultiHop-RAG loaded but is empty: {len(documents)} docs, {len(queries)} queries "
                f"(corpus columns were {corpus_rows.column_names}, "
                f"query columns were {query_rows.column_names})"
            )

        logger.info(
            "Loaded MultiHop-RAG: %d docs, %d queries, %d qrels",
            len(documents),
            len(queries),
            len(judgments),
        )
        return LoadedDataset(documents=documents, queries=queries, judgments=judgments)

    def _build_documents(self, corpus_rows: Any) -> tuple[list[Document], dict[str, str]]:
        """Return the parsed corpus plus a (url|title) -> document_id lookup table."""
        documents: list[Document] = []
        document_id_by_lookup_key: dict[str, str] = {}
        for row in corpus_rows:
            title = row.get("title") or ""
            body = row.get("body") or row.get("text") or ""
            url = row.get("url") or row.get("source") or ""
            document_id = self._namespaced(self._stable_id(url or title))
            documents.append(
                Document(
                    document_id=document_id,
                    source=DatasetSource.MULTIHOP_RAG,
                    title=title,
                    text=body,
                    metadata={
                        "url": url,
                        "source": row.get("source", ""),
                        "published_at": row.get("published_at", ""),
                        "category": row.get("category", ""),
                        "author": row.get("author", ""),
                    },
                )
            )
            document_id_by_lookup_key[self._lookup_key(url, title)] = document_id
        return documents, document_id_by_lookup_key

    @staticmethod
    def _lookup_key(url: str | None, title: str | None) -> str:
        """Evidence entries reference articles by url first, title as a fallback."""
        return (url or "").strip().lower() or (title or "").strip().lower()

    @staticmethod
    def _stable_id(text: str) -> str:
        """Deterministically derive a short id from free text (url/title/question)."""
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _namespaced(raw_id: str) -> str:
        return f"{DatasetSource.MULTIHOP_RAG.value}::{raw_id}"
