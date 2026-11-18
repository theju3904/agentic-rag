"""Learned XGBoost router: the decision function of the agentic `learned_router` mode.

A tiny learned classifier sits on top of the lower retrievers (BM25, Dense,
hybrid+rerank, heuristic-router agentic): it computes cheap per-query features
and picks which pool's top-k to return. The retriever is also named by the
agentic sub-strategy `LEARNED_ROUTER` (historically "P6") so training rows
and historical ledger queries resolve.

The target used at training time is tie-aware: the label for a query is the
pipeline that most clearly beats the agentic branch (P5, as recorded under
`agentic_corrective` in the ledger) on nDCG@10 by more than a margin delta
(default 0.10); otherwise the label stays "agentic_corrective" so the router
defaults to P5 whenever the evidence for overriding is weak.

Features used (all computable at inference time, no gold labels):
  - 12 query-text features (length, question words, digits, year/money markers,
    quoted strings, caps/camelcase patterns) - the same ones used in the
    full-population error analysis script.
  - 4 retrieval-overlap features comparing BM25 top-10, Dense top-10 and
    hybrid+rerank top-1 by document id.
  - A dataset one-hot (optional; present in the training ledger but set to
    all zeros at inference time where the caller does not know the dataset).

A trained model is a single JSON file (XGBoost's native format) accompanied by
`learned_router.json` metadata (feature order, pipeline order, training
delta). This keeps the artifact self-describing and framework-portable.

Validation: 5-fold stratified CV over 4,094 fully-covered queries reports mean
lift of +0.0070 nDCG vs the heuristic router (bootstrap 95% CI [+0.0042,
+0.0097], p < 0.0001). See `report/RESULTS_TECHNICAL.md` (the "P6" learned-router
sections) for the full numbers.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from agentic_rag.domain.retrieval import RetrievalStrategy

if TYPE_CHECKING:
    from agentic_rag.config import Settings

logger = logging.getLogger(__name__)

# These regexes mirror `heuristic_query_router._NUMBER` etc. but are duplicated
# here instead of imported because the heuristic router may be refactored or
# retired independently, and router training must stay stable across refactors.
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_MONEY = re.compile(r"[\$£€]|\b(?:USD|EUR|GBP|CHF|JPY|CNY)\b", re.IGNORECASE)
_ALLCAPS_TOKEN = re.compile(r"\b[A-Z]{2,}\b")
_CAMELCASE_TOKEN = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b")
_CAPITAL_TOKEN = re.compile(r"\b[A-Z][A-Za-z0-9]*\b")
_PURE_NUMERIC_TOKEN = re.compile(r"\b-?\d[\d,]*\.?\d*\b")
_QUOTED = re.compile(r"\"[^\"]+\"|'[^']+'")
_WORD = re.compile(r"[A-Za-z0-9]+")

_QUESTION_WORDS = ("what", "when", "where", "who", "whom", "whose", "why", "how", "which")

# Fixed order so training and inference see the same column indices. Append
# only; never reorder, or every previously trained artifact breaks silently.
TEXT_FEATURE_NAMES: tuple[str, ...] = (
    "len_tokens",
    "len_chars",
    "has_question_word",
    "ends_with_qmark",
    "num_digits",
    "has_year",
    "has_money",
    "num_pure_numeric_tokens",
    "num_allcaps_tokens",
    "num_camelcase_tokens",
    "num_quoted_strings",
    "pct_capital_tokens",
)
OVERLAP_FEATURE_NAMES: tuple[str, ...] = (
    "bm25_dense_top1_same",
    "bm25_dense_jaccard_top5",
    "bm25_dense_jaccard_top10",
    "bm25_rerank_top1_same",
)


def extract_text_features(query_text: str) -> dict[str, float]:
    """Compute the 12 cheap query-text features used by the router."""
    text = query_text or ""
    words = _WORD.findall(text)
    lowered = text.lower()
    capital_tokens = [m.group() for m in _CAPITAL_TOKEN.finditer(text) if len(m.group()) >= 2]
    if capital_tokens and text.strip().startswith(capital_tokens[0]):
        # Avoid treating the sentence-leading capital as a signal (same rule as
        # `HeuristicQueryRouter` for consistency).
        capital_tokens = capital_tokens[1:]
    pct_capital = len(capital_tokens) / len(words) if words else 0.0
    return {
        "len_tokens": float(len(words)),
        "len_chars": float(len(text)),
        "has_question_word": float(any(w in lowered.split() for w in _QUESTION_WORDS)),
        "ends_with_qmark": float(text.strip().endswith("?")),
        "num_digits": float(sum(c.isdigit() for c in text)),
        "has_year": float(bool(_YEAR.search(text))),
        "has_money": float(bool(_MONEY.search(text))),
        "num_pure_numeric_tokens": float(len(_PURE_NUMERIC_TOKEN.findall(text))),
        "num_allcaps_tokens": float(len(_ALLCAPS_TOKEN.findall(text))),
        "num_camelcase_tokens": float(len(_CAMELCASE_TOKEN.findall(text))),
        "num_quoted_strings": float(len(_QUOTED.findall(text))),
        "pct_capital_tokens": float(pct_capital),
    }


def extract_overlap_features(
    bm25_doc_ids: list[str],
    dense_doc_ids: list[str],
    rerank_doc_ids: list[str],
) -> dict[str, float]:
    """Compute the 4 cheap retrieval-overlap features used by the router.

    Document ids are deduplicated by first occurrence (so `bm25_doc_ids` is
    the ordered unique list, not raw chunk-id order). All comparisons are over
    document ids - the project's retrieval metrics are judged at doc granularity
    in every dataset's qrels.
    """
    bm25_top = bm25_doc_ids[:10]
    dense_top = dense_doc_ids[:10]
    rerank_top1 = rerank_doc_ids[:1]
    bm25_top1 = bm25_top[:1]

    def jaccard(a: list[str], b: list[str]) -> float:
        sa, sb = set(a), set(b)
        if not sa and not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    return {
        "bm25_dense_top1_same": float(
            bool(bm25_top1) and bool(dense_top[:1]) and bm25_top1[0] == dense_top[0]
        ),
        "bm25_dense_jaccard_top5": jaccard(bm25_top[:5], dense_top[:5]),
        "bm25_dense_jaccard_top10": jaccard(bm25_top[:10], dense_top[:10]),
        "bm25_rerank_top1_same": float(
            bool(bm25_top1) and bool(rerank_top1) and bm25_top1[0] == rerank_top1[0]
        ),
    }


@dataclass(frozen=True, slots=True)
class LearnedRouterArtifact:
    """A trained-router bundle: the XGBoost model plus its schema metadata."""

    booster: xgboost.Booster  # type: ignore[name-defined]  # noqa: F821
    feature_names: tuple[str, ...]
    pipeline_order: tuple[RetrievalStrategy, ...]
    dataset_vocab: tuple[str, ...]
    training_delta: float


class LearnedRouter:
    """Predicts which retrieval pool to return for a given query.

    This is a thin wrapper around a trained XGBoost multiclass classifier. The
    heavy lifting (feature extraction) is done by free functions above so they
    can be unit-tested without a trained model and reused in the training
    script.
    """

    MODEL_FILENAME = "learned_router.xgb.json"
    META_FILENAME = "learned_router.json"

    def __init__(self, artifact: LearnedRouterArtifact) -> None:
        self._artifact = artifact

    @classmethod
    def load(cls, model_dir: Path) -> LearnedRouter:
        """Load a trained router from `model_dir / learned_router.xgb.json` + meta."""
        import xgboost as xgb  # local import so unit tests of feature code need no xgboost

        meta = json.loads((model_dir / cls.META_FILENAME).read_text(encoding="utf-8"))
        booster = xgb.Booster()
        booster.load_model(str(model_dir / cls.MODEL_FILENAME))
        return cls(
            LearnedRouterArtifact(
                booster=booster,
                feature_names=tuple(meta["feature_names"]),
                pipeline_order=tuple(RetrievalStrategy(s) for s in meta["pipeline_order"]),
                dataset_vocab=tuple(meta.get("dataset_vocab", ())),
                training_delta=float(meta.get("training_delta", 0.10)),
            )
        )

    @property
    def pipeline_order(self) -> tuple[RetrievalStrategy, ...]:
        return self._artifact.pipeline_order

    def predict(
        self,
        query_text: str,
        bm25_doc_ids: list[str],
        dense_doc_ids: list[str],
        rerank_doc_ids: list[str],
        dataset: str | None = None,
    ) -> RetrievalStrategy:
        """Return the pipeline whose top-k the caller should hand back."""
        import numpy as np
        import xgboost as xgb

        feats = {
            **extract_text_features(query_text),
            **extract_overlap_features(bm25_doc_ids, dense_doc_ids, rerank_doc_ids),
        }
        for ds in self._artifact.dataset_vocab:
            feats[f"ds={ds}"] = 1.0 if dataset == ds else 0.0

        row = np.zeros((1, len(self._artifact.feature_names)), dtype=np.float32)
        for i, name in enumerate(self._artifact.feature_names):
            row[0, i] = feats.get(name, 0.0)
        probs = self._artifact.booster.predict(xgb.DMatrix(row))
        cls_idx = int(probs.argmax(axis=1)[0])
        return self._artifact.pipeline_order[cls_idx]


def artifact_dir_from_settings(settings: Settings) -> Path:
    """The directory a trained router is loaded from (and training writes to)."""
    return Path(settings.artifacts_dir) / "router"
