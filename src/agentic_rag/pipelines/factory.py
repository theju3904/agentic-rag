"""Composition root: builds all five `RagPipeline`s from shared infrastructure.

This is the one place in the codebase that wires concrete retrievers, storage
backends, and the LLM client together. Every other module depends on
interfaces (`Retriever`, `VectorIndex`, `LLMClient`), not on this factory.
"""

from __future__ import annotations

import json
import logging

from sqlalchemy.orm import Session

from agentic_rag.agent.agentic_retriever import AgenticRetriever
from agentic_rag.agent.corrective_controller import CorrectiveController
from agentic_rag.agent.learned_router import LearnedRouter, artifact_dir_from_settings
from agentic_rag.agent.learned_router_retriever import LearnedRouterRetriever
from agentic_rag.agent.query_rewriter import QueryRewriter
from agentic_rag.agent.relevance_grader import RelevanceGrader
from agentic_rag.config import Settings
from agentic_rag.domain.document import Chunk
from agentic_rag.domain.retrieval import RetrievalStrategy
from agentic_rag.generation.answer_generator import AnswerGenerator
from agentic_rag.generation.factory import build_llm_client
from agentic_rag.pipelines.base import RagPipeline
from agentic_rag.pipelines.rag_pipeline import RetrieveThenGeneratePipeline
from agentic_rag.retrieval.base import Retriever
from agentic_rag.retrieval.bm25_retriever import Bm25Retriever
from agentic_rag.retrieval.cross_encoder_reranker import (
    RerankerProtocol,
    build_reranker,
)
from agentic_rag.retrieval.dense_retriever import DenseRetriever
from agentic_rag.retrieval.embedding_model import build_embedding_model
from agentic_rag.retrieval.hybrid_retriever import HybridRetriever
from agentic_rag.retrieval.reranked_retriever import RerankedRetriever
from agentic_rag.storage.postgres.repository import ChunkRepository
from agentic_rag.storage.vector_index.base import VectorIndex
from agentic_rag.storage.vector_index.factory import (
    build_vector_index as build_configured_vector_index,
)

logger = logging.getLogger(__name__)

# Uncalibrated fallback for the cross-encoder/ms-marco-MiniLM-L-6-v2 score
# range; run `agentic-rag calibrate` to replace these with thresholds fitted
# on the golden dataset's own sample of relevant/irrelevant passages.
_DEFAULT_RELEVANCE_THRESHOLDS = {
    "confident_relevant_threshold": 0.0,
    "confident_irrelevant_threshold": -5.0,
}


class PipelineFactory:
    """Builds the five `RagPipeline`s that share one embedding model, reranker, and LLM client."""

    def __init__(
        self,
        settings: Settings,
        chunk_repository: ChunkRepository,
        session: Session | None = None,
    ) -> None:
        """
        Args:
            settings: application configuration.
            chunk_repository: used by the dense retriever to hydrate ANN hits into `Chunk`s.
            session: an active SQLAlchemy session; required only if
                `settings.vector_backend == "pgvector"`.
        """
        self._settings = settings
        self._chunk_repository = chunk_repository
        self._session = session
        self._embedding_model = build_embedding_model(
            settings.embedding_model_name, settings.remote_embedding_url
        )
        self._llm_client = build_llm_client(settings)

    def build_vector_index(self) -> VectorIndex:
        """Construct the configured `VectorIndex` backend ("qdrant" or "pgvector").

        Raises:
            ValueError: if `pgvector` is selected without an active session.
        """
        return build_configured_vector_index(self._settings, self._embedding_model, self._session)

    def build_all_pipelines(
        self, all_chunks: list[Chunk], vector_index: VectorIndex
    ) -> dict[RetrievalStrategy, RagPipeline]:
        """Build and return every pipeline, keyed by `RetrievalStrategy`.

        That is the five top-level pipelines (P1-P5), plus the agentic
        sub-strategy registration under `LEARNED_ROUTER` when a trained
        artifact exists. The sub-strategy shares its retriever with the
        agentic pipeline's `learned_router` mode and is registered only so
        `agentic-rag train-router` and historical ledger queries resolve
        (callers should iterate `RetrievalStrategy.top_level_pipelines()`
        rather than the dict's keys).

        Args:
            all_chunks: the full chunk collection, used to build the in-memory BM25 index.
            vector_index: the (already populated) `VectorIndex` to search for dense retrieval.
        """
        answer_generator = AnswerGenerator(self._llm_client)

        sparse_retriever = Bm25Retriever(all_chunks)
        dense_retriever = DenseRetriever(
            self._embedding_model, vector_index, self._chunk_repository
        )
        hybrid_retriever = HybridRetriever(sparse_retriever, dense_retriever)
        reranker = build_reranker(
            self._settings.cross_encoder_model_name, self._settings.remote_rerank_url
        )
        reranked_retriever = RerankedRetriever(hybrid_retriever, reranker)

        # The heuristic router is both a selectable agentic mode and the "agentic"
        # branch the learned router was trained and measured against (training
        # reads the `agentic_corrective` ledger rows this mode produced). Build it
        # explicitly, never via `AGENTIC_MODE`, so the learned router can neither
        # recurse into itself nor silently change its fourth branch with the mode.
        heuristic_retriever = self._build_heuristic_router_retriever(
            sparse_retriever, hybrid_retriever, reranker
        )
        learned_retriever = self._maybe_build_learned_router(
            sparse_retriever=sparse_retriever,
            dense_retriever=dense_retriever,
            reranked_retriever=reranked_retriever,
            agentic_retriever=heuristic_retriever,
        )

        # Resolve the user-facing `agentic_corrective` pipeline from AGENTIC_MODE.
        agentic_retriever = self._select_agentic_retriever(
            heuristic_retriever=heuristic_retriever,
            learned_retriever=learned_retriever,
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
        )

        retrievers_by_strategy = {
            RetrievalStrategy.SPARSE_BM25: sparse_retriever,
            RetrievalStrategy.DENSE: dense_retriever,
            RetrievalStrategy.HYBRID_RRF: hybrid_retriever,
            RetrievalStrategy.HYBRID_RERANKED: reranked_retriever,
            RetrievalStrategy.AGENTIC_CORRECTIVE: agentic_retriever,
        }

        # Register the agentic sub-strategy name `LEARNED_ROUTER` so training
        # (`agentic-rag train-router`) and historical ledger queries resolve.
        # The retriever is the same object the agentic pipeline's
        # `learned_router` mode uses, so this is NOT a sixth top-level pipeline.
        # Public listings and the default `evaluate` iterate only
        # `RetrievalStrategy.top_level_pipelines()`.
        if learned_retriever is not None:
            retrievers_by_strategy[RetrievalStrategy.LEARNED_ROUTER] = learned_retriever

        return {
            strategy: RetrieveThenGeneratePipeline(strategy, retriever, answer_generator)
            for strategy, retriever in retrievers_by_strategy.items()
        }

    def _maybe_build_learned_router(
        self,
        sparse_retriever: Bm25Retriever,
        dense_retriever: DenseRetriever,
        reranked_retriever: RerankedRetriever,
        agentic_retriever: Retriever,
    ) -> LearnedRouterRetriever | None:
        """Build the learned-router retriever iff a trained artifact exists.

        The returned object backs **both** the agentic pipeline's
        `learned_router` mode and the `LEARNED_ROUTER` sub-strategy ledger
        registration. A missing artifact is a soft no-op: the sub-strategy is
        omitted from the registry and the agentic pipeline falls back to its
        `heuristic_router` mode. See [[§27]].
        """
        model_dir = artifact_dir_from_settings(self._settings)
        marker = model_dir / LearnedRouter.MODEL_FILENAME
        if not marker.exists():
            logger.info(
                "No learned router artifact at %s; the LEARNED_ROUTER sub-strategy will "
                "not be registered and AGENTIC_MODE=learned_router will fall back to "
                "heuristic_router. Run `agentic-rag train-router` to enable it.",
                marker,
            )
            return None
        return LearnedRouterRetriever(
            sparse_retriever=sparse_retriever,
            dense_retriever=dense_retriever,
            reranked_retriever=reranked_retriever,
            agentic_retriever=agentic_retriever,
            learned_router=LearnedRouter.load(model_dir),
        )

    def _build_per_doc_agentic_retriever(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: RerankerProtocol,
    ) -> AgenticRetriever:
        """CRAG-style per-document grading retriever ([[§25]]-[[§26]]): the `per_doc` mode."""
        thresholds = self._load_relevance_thresholds()
        return AgenticRetriever(
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
            relevance_grader=RelevanceGrader(**thresholds),
            query_rewriter=QueryRewriter(self._llm_client),
            corrective_controller=CorrectiveController(),
        )

    def _build_heuristic_router_retriever(
        self,
        sparse_retriever: Bm25Retriever,
        hybrid_retriever: HybridRetriever,
        reranker: RerankerProtocol,
    ) -> Retriever:
        """Text-feature router: BM25 (no rerank) vs Hybrid+Rerank, grader kept for abstention.

        The zero-training agentic mode, and the agentic branch of the learned
        router. See [[§26]] for motivation.
        """
        from agentic_rag.agent.heuristic_query_router import HeuristicQueryRouter
        from agentic_rag.agent.router_agentic_retriever import RouterAgenticRetriever

        thresholds = self._load_relevance_thresholds()
        return RouterAgenticRetriever(
            sparse_retriever=sparse_retriever,
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
            query_router=HeuristicQueryRouter(),
            relevance_grader=RelevanceGrader(**thresholds),
        )

    def _select_agentic_retriever(
        self,
        heuristic_retriever: Retriever,
        learned_retriever: LearnedRouterRetriever | None,
        hybrid_retriever: HybridRetriever,
        reranker: RerankerProtocol,
    ) -> Retriever:
        """Pick the agentic variant named by `settings.agentic_mode`.

        All three modes return a concrete `Retriever`; the rest of the factory is
        mode-agnostic. "Agentic" here means a per-query runtime decision function
        over retrieval, whether that function is a trained classifier, a
        hand-written heuristic, or an LLM grader.

          - "learned_router" (RECOMMENDED): XGBoost classifier picks per query
            among BM25 / Dense / Hybrid+Rerank / the heuristic router after
            running them all. Needs the artifact from `agentic-rag train-router`;
            falls back to "heuristic_router" if it is missing.
          - "heuristic_router": see `_build_heuristic_router_retriever`.
          - "per_doc": hybrid+rerank with CRAG-style per-document grading and
            single-retry/abstain. See [[§25]]-[[§26]].
        """
        mode = self._settings.agentic_mode.lower()

        if mode == "learned_router":
            if learned_retriever is not None:
                return learned_retriever
            logger.warning(
                "AGENTIC_MODE=learned_router requested but no artifact found; "
                "falling back to 'heuristic_router'. Run `agentic-rag train-router` "
                "to enable the learned-router mode."
            )
            return heuristic_retriever

        if mode == "heuristic_router":
            return heuristic_retriever

        if mode != "per_doc":
            logger.warning("Unknown AGENTIC_MODE '%s'; falling back to 'per_doc'.", mode)
        return self._build_per_doc_agentic_retriever(hybrid_retriever, reranker)

    def _load_relevance_thresholds(self) -> dict[str, float]:
        thresholds_path = self._settings.artifacts_dir / "agentic_thresholds.json"
        if not thresholds_path.exists():
            logger.warning(
                "No calibrated thresholds at %s; using uncalibrated defaults %s. "
                "Run `agentic-rag calibrate` for calibrated numbers.",
                thresholds_path,
                _DEFAULT_RELEVANCE_THRESHOLDS,
            )
            return dict(_DEFAULT_RELEVANCE_THRESHOLDS)
        return json.loads(thresholds_path.read_text(encoding="utf-8"))
