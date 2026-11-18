"""Stratified sampling utility shared by golden-dataset freezing and
evaluation-time subsampling (see README, "Experimental Design").
"""

from __future__ import annotations

import random
from collections.abc import Callable

from agentic_rag.domain.query import Query


def stratified_sample(
    queries: list[Query], max_total: int, group_key: Callable[[Query], str], random_seed: int
) -> list[Query]:
    """Sample down to at most `max_total` queries, preserving each group's proportion.

    Grouping (rather than a plain random sample) guarantees small/rare groups
    (e.g. a minority query type, or a small dataset) survive the subsample
    instead of being drowned out by the largest group.

    Args:
        queries: the full query list to sample from.
        max_total: the maximum number of queries to return.
        group_key: assigns each query to a stratum, e.g. `lambda q: q.source.value`.
        random_seed: seed for the deterministic shuffle within each group.
    """
    if len(queries) <= max_total:
        return list(queries)

    rng = random.Random(random_seed)
    queries_by_group: dict[str, list[Query]] = {}
    for query in queries:
        queries_by_group.setdefault(group_key(query), []).append(query)

    sample_fraction = max_total / len(queries)
    sampled: list[Query] = []
    for group_queries in queries_by_group.values():
        rng.shuffle(group_queries)
        take = max(1, round(len(group_queries) * sample_fraction))
        sampled.extend(group_queries[:take])
    return sampled
