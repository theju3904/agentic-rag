"""Query entities: the benchmark questions pipelines are evaluated against."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from agentic_rag.domain.document import DatasetSource


class QueryType(StrEnum):
    """Coarse query taxonomy used by the agentic pipeline's routing logic.

    Mirrors the four MultiHop-RAG query types, plus a generic bucket for the
    single-hop BEIR / T2-RAGBench questions that need no special handling.
    """

    SIMPLE = "simple"
    COMPARISON = "comparison"
    TEMPORAL = "temporal"
    INFERENCE = "inference"
    NULL = "null"  # expected to have no answerable evidence in the corpus


@dataclass(frozen=True, slots=True)
class Query:
    """A benchmark question together with the ground truth needed to score it."""

    query_id: str
    source: DatasetSource
    text: str
    query_type: QueryType = QueryType.SIMPLE
    gold_answer: str | None = None
    metadata: dict = field(default_factory=dict)
