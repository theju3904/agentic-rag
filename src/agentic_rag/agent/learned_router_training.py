"""Train the learned-router model (`LEARNED_ROUTER`, historically "P6") from an eval ledger.

Reads one or more `pipeline_run_records` runs out of Postgres, computes nDCG@10
for each (query, strategy) pair against `relevance_judgments`, builds the same
query-text + retrieval-overlap features that `agent/learned_router.py` extracts
at inference time, trains an XGBoost multiclass classifier with a tie-aware
target, and writes the booster plus a metadata JSON to `<artifacts>/router/`.

The training objective
----------------------
Oracle label per query is `argmax_strategy ndcg@10`. We flatten it with a
margin `delta`: unless some strategy beats agentic (P5) by at least `delta`,
the label stays `agentic_corrective`. This lets the router default to P5
whenever the evidence for overriding is weak and matches the behavior
validated in `report/RESULTS_TECHNICAL.md` (the "P6" learned-router sections).

Inputs
------
`run_ids` is a list of pipeline-run ids whose records cover all strategies the
router should learn to arbitrate between. If both baseline and agentic runs
live in the same run, one id is enough; otherwise provide both.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import text as sql_text
from sqlalchemy.engine import Engine

from agentic_rag.agent.learned_router import (
    OVERLAP_FEATURE_NAMES,
    TEXT_FEATURE_NAMES,
    LearnedRouter,
    extract_overlap_features,
    extract_text_features,
)
from agentic_rag.domain.retrieval import RetrievalStrategy
from agentic_rag.evaluation.retrieval_metrics import ndcg_at_k

logger = logging.getLogger(__name__)

# Strategies the router arbitrates between. HYBRID_RRF is omitted on purpose:
# the oracle wins for RRF are tiny (<10%) and highly correlated with
# hybrid+rerank, so including it only confuses the classifier at train time.
_LEARNED_POOL_STRATEGIES: tuple[RetrievalStrategy, ...] = (
    RetrievalStrategy.SPARSE_BM25,
    RetrievalStrategy.DENSE,
    RetrievalStrategy.HYBRID_RERANKED,
    RetrievalStrategy.AGENTIC_CORRECTIVE,
)
_P5_INDEX = _LEARNED_POOL_STRATEGIES.index(RetrievalStrategy.AGENTIC_CORRECTIVE)


@dataclass(slots=True)
class TrainingSample:
    query_id: str
    query_text: str
    dataset: str
    doc_ids_by_strategy: dict[RetrievalStrategy, list[str]]
    ndcg_by_strategy: dict[RetrievalStrategy, float]


def _load_training_samples(
    engine: Engine, run_ids: list[str], top_k: int
) -> list[TrainingSample]:
    """Pull one row per query that has retrievals for every learned pool.

    Rows missing any of the four strategies are skipped - the training target
    needs all of them to decide the oracle label and the model needs their
    doc-id lists to build overlap features.
    """
    judgments: dict[str, dict[str, int]] = defaultdict(dict)
    with engine.connect() as conn:
        for qid, did, rel in conn.execute(sql_text(
            "SELECT query_id, document_id, relevance FROM relevance_judgments"
        )):
            judgments[qid][did] = int(rel)

        query_text_by_id: dict[str, str] = {}
        for qid, qt in conn.execute(sql_text("SELECT query_id, text FROM queries")):
            query_text_by_id[qid] = qt

        retrievals: dict[tuple[str, RetrievalStrategy], tuple[str, list[str]]] = {}
        for qid, dset, strat, dids in conn.execute(sql_text(
            """
            SELECT query_id, dataset, strategy, retrieved_document_ids
            FROM pipeline_run_records
            WHERE run_id = ANY(:rids) AND strategy = ANY(:strats)
            """
        ), {
            "rids": run_ids,
            "strats": [s.value for s in _LEARNED_POOL_STRATEGIES],
        }):
            retrievals[(qid, RetrievalStrategy(strat))] = (dset, list(dids))

    samples: list[TrainingSample] = []
    for qid, qtext in query_text_by_id.items():
        if not all((qid, s) in retrievals for s in _LEARNED_POOL_STRATEGIES):
            continue
        if qid not in judgments:
            continue
        dataset = retrievals[(qid, _LEARNED_POOL_STRATEGIES[0])][0]
        doc_ids_by_strategy = {s: retrievals[(qid, s)][1] for s in _LEARNED_POOL_STRATEGIES}
        ndcg_by_strategy = {
            s: ndcg_at_k(dids, judgments[qid], top_k)
            for s, dids in doc_ids_by_strategy.items()
        }
        samples.append(TrainingSample(
            query_id=qid, query_text=qtext, dataset=dataset,
            doc_ids_by_strategy=doc_ids_by_strategy,
            ndcg_by_strategy=ndcg_by_strategy,
        ))
    return samples


def _tie_aware_label(sample: TrainingSample, delta: float) -> int:
    """Return the class index to use at train time (defaults to P5 on near-ties)."""
    import numpy as np

    scores = np.array([sample.ndcg_by_strategy[s] for s in _LEARNED_POOL_STRATEGIES])
    best = int(np.argmax(scores))
    p5_score = scores[_P5_INDEX]
    if best == _P5_INDEX or (scores[best] - p5_score) < delta:
        return _P5_INDEX
    return best


def train_learned_router(
    engine: Engine,
    run_ids: list[str],
    out_dir: Path,
    *,
    top_k: int = 10,
    delta: float = 0.10,
    n_estimators: int = 400,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> dict[str, float | int]:
    """Train the learned router and write the artifact to `out_dir`.

    Returns a small summary dict with train-set accuracy and class counts so
    callers (CLI) can log something useful without re-reading the artifact.
    """
    import numpy as np
    import xgboost as xgb

    samples = _load_training_samples(engine, run_ids, top_k)
    if not samples:
        raise RuntimeError(
            "No training samples found - every learned pool must have retrievals "
            "for the same queries, and relevance_judgments must cover them."
        )
    logger.info("Loaded %d training samples (queries with all 4 pools)", len(samples))

    dataset_vocab = tuple(sorted({s.dataset for s in samples}))
    feature_names = tuple(
        list(TEXT_FEATURE_NAMES) + list(OVERLAP_FEATURE_NAMES) + [f"ds={d}" for d in dataset_vocab]
    )
    X = np.zeros((len(samples), len(feature_names)), dtype=np.float32)
    y = np.zeros(len(samples), dtype=np.int32)
    bm25_s = RetrievalStrategy.SPARSE_BM25
    dense_s = RetrievalStrategy.DENSE
    rerank_s = RetrievalStrategy.HYBRID_RERANKED
    for i, sample in enumerate(samples):
        text_f = extract_text_features(sample.query_text)
        overlap_f = extract_overlap_features(
            sample.doc_ids_by_strategy[bm25_s],
            sample.doc_ids_by_strategy[dense_s],
            sample.doc_ids_by_strategy[rerank_s],
        )
        for j, name in enumerate(feature_names):
            if name.startswith("ds="):
                X[i, j] = 1.0 if name == f"ds={sample.dataset}" else 0.0
            elif name in text_f:
                X[i, j] = text_f[name]
            else:
                X[i, j] = overlap_f[name]
        y[i] = _tie_aware_label(sample, delta)

    label_counts = {
        int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True), strict=True)
    }
    logger.info("Tie-aware label distribution (delta=%.2f): %s", delta, label_counts)

    model = xgb.XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=len(_LEARNED_POOL_STRATEGIES),
        random_state=random_state, eval_metric="mlogloss", n_jobs=-1, tree_method="hist",
    )
    model.fit(X, y)
    train_acc = float((model.predict(X) == y).mean())
    logger.info("Train accuracy: %.4f", train_acc)

    out_dir.mkdir(parents=True, exist_ok=True)
    booster = model.get_booster()
    booster.save_model(str(out_dir / LearnedRouter.MODEL_FILENAME))
    (out_dir / LearnedRouter.META_FILENAME).write_text(
        json.dumps({
            "feature_names": list(feature_names),
            "pipeline_order": [s.value for s in _LEARNED_POOL_STRATEGIES],
            "dataset_vocab": list(dataset_vocab),
            "training_delta": delta,
            "n_training_queries": len(samples),
            "label_distribution": {
                _LEARNED_POOL_STRATEGIES[k].value: v for k, v in label_counts.items()
            },
        }, indent=2),
        encoding="utf-8",
    )
    return {
        "n_training_queries": len(samples),
        "train_accuracy": train_acc,
        **{f"label_{_LEARNED_POOL_STRATEGIES[k].value}": v for k, v in label_counts.items()},
    }
