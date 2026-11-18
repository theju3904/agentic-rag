"""Builds the frozen "golden dataset" used for every experiment in this project.

Combines BEIR (SciFact, NFCorpus, FiQA), MultiHop-RAG, and T2-RAGBench into one
corpus/query/qrels triple. The two largest query sets are stratified-sampled
down to a fixed cap so the expensive rerank/agentic pipelines stay tractable
on a single CPU machine; every other dataset is kept whole. See README,
"Experimental Design", for the full rationale and the exact sample sizes used.
"""

from __future__ import annotations

import logging

from agentic_rag.data.base_loader import DatasetLoader, LoadedDataset
from agentic_rag.domain.document import DatasetSource, Document
from agentic_rag.domain.evaluation import RelevanceJudgment
from agentic_rag.domain.query import Query
from agentic_rag.evaluation.sampling import stratified_sample

logger = logging.getLogger(__name__)

# Datasets larger than this many queries are stratified-subsampled (by query
# type) down to this size before being frozen to disk.
_MAX_QUERIES_PER_LARGE_DATASET = 1_500
_LARGE_DATASET_SOURCES = {DatasetSource.T2_RAGBENCH, DatasetSource.MULTIHOP_RAG}


class GoldenDatasetBuilder:
    """Combines multiple `DatasetLoader`s into one frozen, combined dataset."""

    def __init__(self, loaders: dict[DatasetSource, DatasetLoader], random_seed: int) -> None:
        """
        Args:
            loaders: one `DatasetLoader` per dataset to include in the golden set.
            random_seed: seed for the deterministic stratified subsampling step.
        """
        self._loaders = loaders
        self._random_seed = random_seed

    def build(self) -> LoadedDataset:
        """Load every configured dataset and concatenate them into one.

        Raises:
            DatasetLoadError: propagated from the first loader that fails.
        """
        all_documents: list[Document] = []
        all_queries: list[Query] = []
        all_judgments: list[RelevanceJudgment] = []

        for source, loader in self._loaders.items():
            loaded = loader.load()
            queries, judgments = self._subsample_if_large(source, loaded.queries, loaded.judgments)
            all_documents.extend(loaded.documents)
            all_queries.extend(queries)
            all_judgments.extend(judgments)
            logger.info(
                "Golden dataset += %s: %d docs, %d queries (of %d loaded), %d qrels",
                source.value,
                len(loaded.documents),
                len(queries),
                len(loaded.queries),
                len(judgments),
            )

        return LoadedDataset(documents=all_documents, queries=all_queries, judgments=all_judgments)

    def _subsample_if_large(
        self,
        source: DatasetSource,
        queries: list[Query],
        judgments: list[RelevanceJudgment],
    ) -> tuple[list[Query], list[RelevanceJudgment]]:
        """Stratify-sample a dataset's queries by `query_type` down to a fixed cap.

        Stratifying (rather than a plain random sample) guarantees rare query
        types, such as MultiHop-RAG's "null" abstention queries, survive the
        subsample instead of being drowned out by the majority type.
        """
        if source not in _LARGE_DATASET_SOURCES or len(queries) <= _MAX_QUERIES_PER_LARGE_DATASET:
            return queries, judgments

        sampled_queries = stratified_sample(
            queries,
            max_total=_MAX_QUERIES_PER_LARGE_DATASET,
            group_key=lambda query: query.query_type.value,
            random_seed=self._random_seed,
        )
        sampled_query_ids = {query.query_id for query in sampled_queries}
        sampled_judgments = [j for j in judgments if j.query_id in sampled_query_ids]
        return sampled_queries, sampled_judgments
