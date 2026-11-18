"""Regenerate the dashboard/report figures from the evaluation numbers.

Run:  python report/figures/make_figures.py
Writes PNGs into report/figures/ (committed, referenced by the results docs).

Experiment A numbers come from the full 4,271-query run (report/results.md).
Experiment B numbers come from the fair retrieval-only re-run (same full query
set, three embedding models) - filled in from the generated per-model reports.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "figure.dpi": 140,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)

DATASETS = ["scifact", "nfcorpus", "fiqa", "multihop_rag", "t2_ragbench"]
DS_LABEL = {
    "scifact": "SciFact",
    "nfcorpus": "NFCorpus",
    "fiqa": "FiQA",
    "multihop_rag": "MultiHop",
    "t2_ragbench": "T2-RAGBench",
}
PIPE_COLOR = {
    "sparse_bm25": "#d97706",
    "dense": "#2563eb",
    "hybrid_rrf": "#059669",
    "hybrid_reranked": "#7c3aed",
    "agentic_corrective": "#dc2626",
}
PIPE_LABEL = {
    "sparse_bm25": "BM25 (sparse)",
    "dense": "Dense",
    "hybrid_rrf": "Hybrid RRF",
    "hybrid_reranked": "Hybrid+Rerank",
    "agentic_corrective": "Agentic",
}
MODEL_COLOR = {"minilm": "#2563eb", "bge_m3": "#7c3aed", "bge_large": "#059669"}
MODEL_LABEL = {
    "minilm": "MiniLM-L6 (384d)",
    "bge_m3": "bge-m3 (1024d)",
    "bge_large": "bge-large (1024d)",
}

# ── Experiment A: full 4,271-query run, nDCG@10 (report/results.md, Tier 1) ──────
EXP_A_NDCG = {
    "sparse_bm25": {"scifact": .6226, "nfcorpus": .2311, "fiqa": .2058, "multihop_rag": .5352, "t2_ragbench": .5352},
    "dense":       {"scifact": .6147, "nfcorpus": .2346, "fiqa": .3640, "multihop_rag": .4394, "t2_ragbench": .2020},
    "hybrid_rrf":  {"scifact": .6649, "nfcorpus": .2564, "fiqa": .3333, "multihop_rag": .5256, "t2_ragbench": .4048},
}
# Mean retrieval latency (s/query), Experiment A (report/results.md), avg across datasets
EXP_A_LATENCY = {
    "dense": 0.033, "hybrid_rrf": 0.90, "sparse_bm25": 0.95,
    "hybrid_reranked": 6.6, "agentic_corrective": 6.8,
}

# ── Experiment B: fair dense comparison, same full 4,271-query set, 3 models ──────
# DENSE nDCG@10 per dataset (isolates embedding quality). All three produced by the
# SAME batched scoring code (scratchpad/dense_compare.py) against the full query set
# - identical queries, identical metric, so the only variable is the embedding model.
EXP_B_DENSE_NDCG = {
    "minilm":    {"scifact": .6211, "nfcorpus": .2353, "fiqa": .3631, "multihop_rag": .4391, "t2_ragbench": .2054},
    "bge_m3":    {"scifact": .6047, "nfcorpus": .2466, "fiqa": .3892, "multihop_rag": .4986, "t2_ragbench": .4440},
    "bge_large": {"scifact": .7112, "nfcorpus": .2998, "fiqa": .4345, "multihop_rag": .4447, "t2_ragbench": .3627},
}


def fig_mean_ndcg_pipelines() -> None:
    """A1: mean nDCG@10 across datasets for the 3 full-set pipelines."""
    pipes = ["dense", "sparse_bm25", "hybrid_rrf"]
    means = [sum(EXP_A_NDCG[p].values()) / len(DATASETS) for p in pipes]
    order = sorted(range(len(pipes)), key=lambda i: means[i])
    pipes = [pipes[i] for i in order]
    means = [means[i] for i in order]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bars = ax.barh([PIPE_LABEL[p] for p in pipes], means,
                   color=[PIPE_COLOR[p] for p in pipes], height=0.6)
    for b, m in zip(bars, means):
        ax.text(m + 0.005, b.get_y() + b.get_height() / 2, f"{m:.3f}",
                va="center", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("Mean nDCG@10 (averaged over 5 datasets, equal weight)")
    ax.set_title("Which retrieval strategy wins on average?\nHybrid RRF leads (full 4,271-query run)",
                 fontsize=12, loc="left")
    fig.savefig(OUT / "a1_mean_ndcg_pipelines.png")
    plt.close(fig)


def fig_ndcg_by_dataset() -> None:
    """A2: grouped nDCG@10 per dataset for the 3 full-set pipelines."""
    pipes = ["sparse_bm25", "dense", "hybrid_rrf"]
    x = range(len(DATASETS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for j, p in enumerate(pipes):
        vals = [EXP_A_NDCG[p][d] for d in DATASETS]
        ax.bar([i + (j - 1) * w for i in x], vals, w,
               label=PIPE_LABEL[p], color=PIPE_COLOR[p])
    ax.set_xticks(list(x))
    ax.set_xticklabels([DS_LABEL[d] for d in DATASETS])
    ax.set_ylabel("nDCG@10")
    ax.set_ylim(0, 0.75)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.set_title("No single strategy wins everywhere (full 4,271-query run)",
                 fontsize=12, loc="left", pad=28)
    # annotate per-dataset winner
    for i, d in enumerate(DATASETS):
        best = max(pipes, key=lambda p: EXP_A_NDCG[p][d])
        ax.scatter(i + (pipes.index(best) - 1) * w, EXP_A_NDCG[best][d] + 0.02,
                   marker="*", s=90, color="#111", zorder=5)
    fig.savefig(OUT / "a2_ndcg_by_dataset.png")
    plt.close(fig)


def fig_latency() -> None:
    """A3: retrieval latency by pipeline (log scale)."""
    pipes = ["dense", "hybrid_rrf", "sparse_bm25", "hybrid_reranked", "agentic_corrective"]
    vals = [EXP_A_LATENCY[p] for p in pipes]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = ax.bar([PIPE_LABEL[p] for p in pipes], vals,
                  color=[PIPE_COLOR[p] for p in pipes])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.08, f"{v:g}s",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylabel("Mean retrieval latency (s/query, log scale)")
    ax.set_title("Cost of each strategy spans ~200x\n(dense << sparse/hybrid << cross-encoder)",
                 fontsize=12, loc="left")
    fig.savefig(OUT / "a3_latency.png")
    plt.close(fig)


# All five pipelines, nDCG@10, FULL 4,271-query set (GPU reranking).
# Baselines from run-20261008T012154-2185a5b5 (BM25/Dense/Hybrid-RRF/Hybrid+Rerank).
# Agentic row uses the recommended `learned_router` mode from run-20261009T025041-e5cd6aa4
# (XGBoost over 16 features, tie-aware δ=0.10). All numbers recomputed via
# scratchpad/full_picture.py directly against pipeline_run_records + relevance_judgments,
# so they are the single source of truth for the chart.
# n per dataset: scifact 300, nfcorpus 323, fiqa 648, multihop 1500, t2 1500.
EXP_A5_NDCG = {
    "sparse_bm25":        {"scifact": .6226, "nfcorpus": .2311, "fiqa": .2058, "multihop_rag": .6068, "t2_ragbench": .5346},
    "dense":              {"scifact": .6211, "nfcorpus": .2349, "fiqa": .3631, "multihop_rag": .4979, "t2_ragbench": .2038},
    "hybrid_rrf":         {"scifact": .6691, "nfcorpus": .2568, "fiqa": .3334, "multihop_rag": .5951, "t2_ragbench": .4054},
    "hybrid_reranked":    {"scifact": .6841, "nfcorpus": .2812, "fiqa": .3664, "multihop_rag": .5481, "t2_ragbench": .4090},
    "agentic_corrective": {"scifact": .7151, "nfcorpus": .2920, "fiqa": .3712, "multihop_rag": .6489, "t2_ragbench": .5794},
}


def fig_all5_by_dataset() -> None:
    """All five pipelines (incl. reranked + agentic) per dataset, one shared query set.

    A black star marks the winner per dataset.
    """
    pipes = ["sparse_bm25", "dense", "hybrid_rrf", "hybrid_reranked", "agentic_corrective"]
    x = range(len(DATASETS))
    w = 0.16
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    for j, p in enumerate(pipes):
        vals = [EXP_A5_NDCG[p][d] for d in DATASETS]
        ax.bar([i + (j - 2) * w for i in x], vals, w, label=PIPE_LABEL[p], color=PIPE_COLOR[p])
    # Winner star per dataset.
    for i, d in enumerate(DATASETS):
        best = max(pipes, key=lambda p: EXP_A5_NDCG[p][d])
        ax.scatter(
            i + (pipes.index(best) - 2) * w,
            EXP_A5_NDCG[best][d] + 0.02,
            marker="*", s=120, color="#111", zorder=5,
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels([DS_LABEL[d] for d in DATASETS])
    ax.set_ylabel("nDCG@10")
    ax.set_ylim(0, 0.8)
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=9)
    ax.set_title(
        "Agentic pipeline (learned_router mode) wins every dataset on the full 4,271-query set\n"
        "(star = winner per dataset)",
        fontsize=12, loc="left", pad=30,
    )
    fig.savefig(OUT / "a4_all5_by_dataset.png")
    plt.close(fig)


def fig_all5_mean() -> None:
    """Mean nDCG@10 across datasets for all five pipelines."""
    pipes = ["dense", "sparse_bm25", "hybrid_rrf", "agentic_corrective", "hybrid_reranked"]
    means = [sum(EXP_A5_NDCG[p].values()) / len(DATASETS) for p in pipes]
    order = sorted(range(len(pipes)), key=lambda i: means[i])
    pipes = [pipes[i] for i in order]
    means = [means[i] for i in order]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    bars = ax.barh([PIPE_LABEL[p] for p in pipes], means,
                   color=[PIPE_COLOR[p] for p in pipes], height=0.62)
    for b, m in zip(bars, means):
        ax.text(m + 0.005, b.get_y() + b.get_height() / 2, f"{m:.3f}",
                va="center", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 0.6)
    ax.set_xlabel("Mean nDCG@10 (5 datasets, same shared query set)")
    ax.set_title("Mean nDCG@10 across datasets: agentic (learned_router) leads by +0.063",
                 fontsize=12, loc="left")
    fig.savefig(OUT / "a5_all5_mean.png")
    plt.close(fig)


def _grouped_model_chart(data: dict, title: str, fname: str, ylabel: str) -> None:
    """Grouped bars per dataset across embedding models; star marks winner per dataset."""
    models = ["minilm", "bge_m3", "bge_large"]
    if any(not data[m] for m in models):
        print(f"skip {fname}: Experiment B data not filled yet")
        return
    x = range(len(DATASETS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for j, m in enumerate(models):
        vals = [data[m][d] for d in DATASETS]
        ax.bar([i + (j - 1) * w for i in x], vals, w,
               label=MODEL_LABEL[m], color=MODEL_COLOR[m])
    for i, d in enumerate(DATASETS):
        best = max(models, key=lambda m: data[m][d])
        ax.scatter(
            i + (models.index(best) - 1) * w,
            data[best][d] + 0.02,
            marker="*", s=110, color="#111", zorder=5,
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels([DS_LABEL[d] for d in DATASETS])
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 0.8)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.set_title(title + " (star = winner per dataset)", fontsize=12, loc="left", pad=28)
    fig.savefig(OUT / fname)
    plt.close(fig)


def fig_embedding_dense() -> None:
    """B1: dense nDCG@10 per dataset, 3 embedding models (fair, same query set)."""
    _grouped_model_chart(
        EXP_B_DENSE_NDCG,
        "Does a bigger embedding model help? Dense retrieval, same 4,271 queries",
        "b1_embedding_dense_ndcg.png",
        "Dense nDCG@10",
    )


def fig_embedding_mean() -> None:
    """B2: mean dense nDCG@10 across datasets, 3 embedding models."""
    models = ["minilm", "bge_m3", "bge_large"]
    if any(not EXP_B_DENSE_NDCG[m] for m in models):
        print("skip b2: Experiment B data not filled yet")
        return
    means = [sum(EXP_B_DENSE_NDCG[m].values()) / len(DATASETS) for m in models]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    bars = ax.bar([MODEL_LABEL[m] for m in models], means,
                  color=[MODEL_COLOR[m] for m in models])
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.005, f"{m:.3f}",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 0.5)
    ax.set_ylabel("Mean dense nDCG@10 (5 datasets)")
    ax.set_title("Embedding model: mean dense retrieval quality", fontsize=12, loc="left")
    fig.savefig(OUT / "b2_embedding_mean_ndcg.png")
    plt.close(fig)


def fig_per_query_delta() -> None:
    """Per-query scatter: reranked nDCG vs new_p5 nDCG, dot per query, coloured by dataset.

    Points above the diagonal = agentic wins that query. Below = reranked wins.
    Reads the DB directly for the 4,271 per-query scores so the chart always
    reflects the current ledger numbers.
    """
    import math
    import json as _json
    from collections import defaultdict as _dd
    from sqlalchemy import create_engine, text
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))
    from agentic_rag.config import get_settings
    s = get_settings()
    e = create_engine(s.postgres_dsn, pool_timeout=10)
    with e.connect() as c:
        jr = c.execute(text("SELECT query_id, document_id, relevance FROM relevance_judgments")).fetchall()
        J = _dd(dict)
        for q, d, r in jr: J[q][d] = r
        qrows = c.execute(text("SELECT query_id, source FROM queries")).fetchall()
        q_ds = {q: ds for q, ds in qrows}
        def load(rid, strat):
            rows = c.execute(text("SELECT query_id, retrieved_document_ids FROM pipeline_run_records WHERE run_id=:r AND strategy=:s"), {"r": rid, "s": strat}).fetchall()
            return {q: (r if isinstance(r, list) else _json.loads(r)) for q, r in rows}
        new_p5 = load("run-20261008T102911-4f7e29fe", "agentic_corrective")
        rer = load("run-20261008T012154-2185a5b5", "hybrid_reranked")
    def ndcg(ret, judg, k=10):
        if not judg: return 0.0
        dcg = sum(judg.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ret[:k]))
        ideal = sorted(judg.values(), reverse=True)[:k]
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
        return dcg / idcg if idcg > 0 else 0.0
    pts = []
    for qid in set(new_p5) & set(rer):
        pts.append((ndcg(rer[qid], J.get(qid, {})), ndcg(new_p5[qid], J.get(qid, {})), q_ds.get(qid, "?")))
    import random as _r
    _r.Random(42).shuffle(pts)  # z-order variety

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ds_order = ["t2_ragbench", "multihop_rag", "fiqa", "nfcorpus", "scifact"]
    ds_color = {"t2_ragbench": "#dc2626", "multihop_rag": "#d97706", "fiqa": "#059669", "nfcorpus": "#2563eb", "scifact": "#7c3aed"}
    for ds in ds_order:
        xs = [p[0] for p in pts if p[2] == ds]
        ys = [p[1] for p in pts if p[2] == ds]
        ax.scatter(xs, ys, s=10, alpha=0.45, color=ds_color[ds], label=DS_LABEL[ds], edgecolors="none")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="equal")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal")
    ax.set_xlabel("Reranked (P4) nDCG@10 per query")
    ax.set_ylabel("Agentic (heuristic router) nDCG@10 per query")
    ax.legend(frameon=False, loc="upper left", fontsize=9, markerscale=2)
    ax.set_title("Per-query nDCG@10, agentic vs reranked (above-diagonal = agent wins)",
                 fontsize=12, loc="left")
    # subtle shading of the agent-wins region
    ax.fill_between([0, 1], [0, 1], [1.02, 1.02], alpha=0.03, color="green")
    fig.savefig(OUT / "a6_per_query_delta.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_mean_ndcg_pipelines()
    fig_ndcg_by_dataset()
    fig_latency()
    fig_all5_by_dataset()
    fig_all5_mean()
    fig_embedding_dense()
    fig_embedding_mean()
    fig_per_query_delta()
    print("Figures written to", OUT)
