"""The agentic pipeline (P5): per-query, routing-aware retrieval.

Unlike P1-P4, which always execute the same fixed sequence of stages, the
agentic pipeline makes a runtime decision per query. It has three modes
(`AGENTIC_MODE`), differing only in the decision function:

  - `learned_router`: a trained XGBoost classifier picks a retrieval pool
    (recommended default; falls back to `heuristic_router` without an artifact).
  - `heuristic_router`: hand-written rules route to BM25 or hybrid+rerank.
  - `per_doc`: a LangGraph `StateGraph` that grades retrieval quality and
    decides whether to proceed, rewrite the query and retry, or abstain (CRAG).

See README, "The pipelines", for the design rationale.
"""
