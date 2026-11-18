"""Cross-encoder reranking stage shared by P4 (hybrid+rerank) and P5 (agentic).

Two implementations behind one `rerank()` contract: a local `CrossEncoder`
(CPU/GPU in-process) and a `RemoteCrossEncoderReranker` that offloads scoring to
a GPU HTTP server. Both return identical scores for the same model in fp32, so
the agentic pipeline's calibrated thresholds hold either way.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Protocol, runtime_checkable

from agentic_rag.domain.retrieval import ScoredChunk

logger = logging.getLogger(__name__)


@runtime_checkable
class RerankerProtocol(Protocol):
    """Shared interface for local and remote cross-encoder rerankers.

    Both `CrossEncoderReranker` (sentence-transformers in-process) and
    `RemoteCrossEncoderReranker` (GPU HTTP server) satisfy this structurally.
    Returned scores are raw cross-encoder logits, comparable only within one
    query's pool; the agentic pipeline's calibrated thresholds are tied to
    these scores, so both backends must run the same model in fp32.
    """

    def rerank(
        self, query_text: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]: ...


class CrossEncoderReranker:
    """Re-scores a candidate pool with a cross-encoder, jointly encoding query+chunk.

    Cross-encoder scores are NOT comparable across queries or across models,
    so this class only ever reorders one query's own candidate pool; callers
    must not mix its scores with any other stage's scores without first
    converting to rank (research doc, "Controls checklist").
    """

    def __init__(self, model_name: str, max_sequence_length: int = 512) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name, max_length=max_sequence_length)

    def rerank(
        self, query_text: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        """Return the `top_k` candidates re-sorted by cross-encoder relevance score."""
        if not candidates:
            return []
        pairs = [(query_text, candidate.chunk.text) for candidate in candidates]
        raw_scores = self._model.predict(pairs)

        rescored = [
            ScoredChunk(
                chunk=candidate.chunk,
                score=float(score),
                rank=0,  # placeholder, overwritten below once final order is known
                stage_scores={**candidate.stage_scores, "cross_encoder": float(score)},
            )
            for candidate, score in zip(candidates, raw_scores, strict=True)
        ]
        rescored.sort(key=lambda sc: sc.score, reverse=True)
        top_candidates = rescored[:top_k]
        return [
            ScoredChunk(chunk=sc.chunk, score=sc.score, rank=i, stage_scores=sc.stage_scores)
            for i, sc in enumerate(top_candidates)
        ]


class RemoteCrossEncoderReranker:
    """Offloads cross-encoder scoring to a GPU HTTP server (same `rerank()` contract).

    The server must expose POST /rerank {"query": str, "passages": [str, ...]}
    returning {"scores": [float, ...]} in the same order, running the same model
    in fp32 so scores match the local `CrossEncoderReranker` exactly.
    """

    def __init__(self, base_url: str, model_name: str) -> None:
        self._url = f"{base_url.rstrip('/')}/rerank"
        self._model_name = model_name
        logger.info("RemoteCrossEncoderReranker: %s @ %s", model_name, base_url)

    def _score(self, query_text: str, passages: list[str]) -> list[float]:
        payload = json.dumps(
            {"query": query_text, "passages": passages, "model": self._model_name}
        ).encode()
        req = urllib.request.Request(
            self._url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
        )
        last_exc: Exception | None = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    return json.loads(resp.read())["scores"]
            except (urllib.error.URLError, OSError) as exc:
                last_exc = exc
                if attempt < 3:
                    time.sleep(2**attempt)
        raise last_exc  # type: ignore[misc]

    def rerank(
        self, query_text: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        scores = self._score(query_text, [c.chunk.text for c in candidates])
        rescored = [
            ScoredChunk(
                chunk=candidate.chunk,
                score=float(score),
                rank=0,
                stage_scores={**candidate.stage_scores, "cross_encoder": float(score)},
            )
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        rescored.sort(key=lambda sc: sc.score, reverse=True)
        return [
            ScoredChunk(chunk=sc.chunk, score=sc.score, rank=i, stage_scores=sc.stage_scores)
            for i, sc in enumerate(rescored[:top_k])
        ]


def build_reranker(
    model_name: str, remote_url: str | None = None
) -> RerankerProtocol:
    """Return a local or remote reranker depending on whether a URL is set."""
    if remote_url:
        return RemoteCrossEncoderReranker(base_url=remote_url, model_name=model_name)
    return CrossEncoderReranker(model_name)
