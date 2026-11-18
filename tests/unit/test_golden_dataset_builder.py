"""Tests for the stratified subsampling used to bound large datasets' query counts."""

from __future__ import annotations

from agentic_rag.data.base_loader import DatasetLoader, LoadedDataset
from agentic_rag.data.golden_dataset_builder import GoldenDatasetBuilder
from agentic_rag.domain.document import DatasetSource
from agentic_rag.domain.evaluation import RelevanceJudgment
from agentic_rag.domain.query import Query, QueryType


class _FakeLoader(DatasetLoader):
    def __init__(self, dataset: LoadedDataset) -> None:
        self._dataset = dataset

    def load(self) -> LoadedDataset:
        return self._dataset


def _make_query(index: int, source: DatasetSource, query_type: QueryType) -> Query:
    return Query(
        query_id=f"q{index}", source=source, text=f"question {index}", query_type=query_type
    )


def test_small_dataset_is_not_subsampled():
    queries = [_make_query(i, DatasetSource.SCIFACT, QueryType.SIMPLE) for i in range(10)]
    dataset = LoadedDataset(documents=[], queries=queries, judgments=[])
    builder = GoldenDatasetBuilder({DatasetSource.SCIFACT: _FakeLoader(dataset)}, random_seed=42)

    combined = builder.build()

    assert len(combined.queries) == 10


def test_large_dataset_is_stratify_sampled_and_keeps_rare_types():
    # 2000 "simple" queries plus 5 "null" queries: the rare null queries must
    # survive the subsample even though they're a tiny minority of the data.
    simple_queries = [
        _make_query(i, DatasetSource.MULTIHOP_RAG, QueryType.SIMPLE) for i in range(2000)
    ]
    null_queries = [
        _make_query(i, DatasetSource.MULTIHOP_RAG, QueryType.NULL) for i in range(2000, 2005)
    ]
    all_queries = simple_queries + null_queries
    judgments = [
        RelevanceJudgment(query_id=query.query_id, document_id="doc1", relevance=1)
        for query in all_queries
    ]
    dataset = LoadedDataset(documents=[], queries=all_queries, judgments=judgments)
    loaders = {DatasetSource.MULTIHOP_RAG: _FakeLoader(dataset)}
    builder = GoldenDatasetBuilder(loaders, random_seed=42)

    combined = builder.build()

    assert len(combined.queries) <= 1500
    assert QueryType.NULL in {query.query_type for query in combined.queries}

    sampled_query_ids = {query.query_id for query in combined.queries}
    assert all(judgment.query_id in sampled_query_ids for judgment in combined.judgments)
