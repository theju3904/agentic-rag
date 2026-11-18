"""Orchestrates a full evaluation run: every pipeline, over every query,
persisted to the Postgres ledger and summarized into per-dataset metrics.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass

from agentic_rag.domain.evaluation import PipelineRunRecord
from agentic_rag.domain.query import Query
from agentic_rag.domain.retrieval import RetrievalStrategy
from agentic_rag.evaluation.generation_metrics import (
    abstention_accuracy,
    exact_match,
    llm_judge_score,
    numeric_match,
    token_f1,
)
from agentic_rag.evaluation.retrieval_metrics import (
    RetrievalMetricsSummary,
    summarize_retrieval_metrics,
)
from agentic_rag.generation.llm_client import LLMClient
from agentic_rag.pipelines.base import RagPipeline
from agentic_rag.storage.postgres.repository import EvaluationRepository

logger = logging.getLogger(__name__)

_RETRIEVAL_CUTOFF_K = 10
_NUMERIC_ANSWER_DATASET = "t2_ragbench"
_JUDGE_CONTEXT_CHUNKS = 5


@dataclass(frozen=True, slots=True)
class DatasetEvaluationReport:
    """Everything computed for one (dataset, strategy) pair in a run."""

    dataset: str
    strategy: RetrievalStrategy
    retrieval: RetrievalMetricsSummary
    mean_answer_score: float | None
    mean_judge_score: float | None
    abstention_accuracy: float | None
    num_queries: int


class EvaluationRunner:
    """Runs every configured pipeline over every query and aggregates metrics."""

    def __init__(
        self,
        evaluation_repository: EvaluationRepository,
        judgments_by_query: dict[str, dict[str, int]],
        run_top_k: int = _RETRIEVAL_CUTOFF_K,
        judge_llm_client: LLMClient | None = None,
    ) -> None:
        self._evaluation_repository = evaluation_repository
        self._judgments_by_query = judgments_by_query
        self._run_top_k = run_top_k
        self._judge_llm_client = judge_llm_client

    def run(
        self,
        run_id: str,
        pipelines: dict[RetrievalStrategy, RagPipeline],
        queries: list[Query],
        config_snapshot: dict,
    ) -> list[DatasetEvaluationReport]:
        """Start a new run, execute `pipelines` over `queries`, and finish it.

        For a single-pass evaluation. To run different pipelines over
        different query sets under one run id (see README, "Experimental
        Design" -- full query set for the cheap pipelines, a stratified
        subsample for the rerank/agentic ones), call `start`, `run_pass`
        (possibly more than once), and `finish` directly instead.
        """
        self.start(run_id, config_snapshot)
        reports = self.run_pass(run_id, pipelines, queries)
        self.finish(run_id)
        return reports

    def start(self, run_id: str, config_snapshot: dict) -> None:
        """Create the evaluation run's ledger entry. Call once per run id."""
        self._evaluation_repository.start_run(run_id, config_snapshot)

    def finish(self, run_id: str) -> None:
        """Mark the evaluation run's ledger entry as finished. Call once per run id."""
        self._evaluation_repository.finish_run(run_id)

    def run_pass(
        self,
        run_id: str,
        pipelines: dict[RetrievalStrategy, RagPipeline],
        queries: list[Query],
        generate_answers: bool = True,
        skip_keys: set[tuple[str, str]] | None = None,
    ) -> list[DatasetEvaluationReport]:
        """Execute `pipelines` over `queries` and persist every record.

        Does not start or finish the run; callers doing a multi-pass
        evaluation (different pipelines/queries per pass) call this once per
        pass, between their own `start`/`finish` calls.

        Args:
            generate_answers: if `False`, only retrieval runs -- the LLM is
                never called and generation-quality metrics are omitted from
                the resulting report. Use this for large query sets where
                only Recall/nDCG/MRR/MAP are needed (see README, "Experimental
                design"): retrieval metrics don't require a generated answer.
            skip_keys: set of `(strategy_value, query_id)` pairs already in
                the DB for this run; those pairs are skipped. Used by the
                CLI's resume path (`--append-to`) so a long eval can be split
                across multiple background runs without redoing work.
        """
        queries_by_dataset = self._group_by_dataset(queries)
        skip = skip_keys or set()
        return [
            self._run_one_pipeline_on_one_dataset(
                run_id, strategy, pipeline, dataset_name, dataset_queries, generate_answers,
                self._judge_llm_client, skip_keys=skip,
            )
            for strategy, pipeline in pipelines.items()
            for dataset_name, dataset_queries in queries_by_dataset.items()
        ]

    def _run_one_pipeline_on_one_dataset(
        self,
        run_id: str,
        strategy: RetrievalStrategy,
        pipeline: RagPipeline,
        dataset_name: str,
        queries: list[Query],
        generate_answers: bool,
        judge_llm_client: LLMClient | None = None,
        skip_keys: set[tuple[str, str]] | None = None,
    ) -> DatasetEvaluationReport:
        retrieved_document_ids_by_query: dict[str, list[str]] = {}
        answer_scores: list[float] = []
        judge_scores: list[float] = []
        abstention_predictions: list[bool] = []
        abstention_gold: list[bool] = []
        skip = skip_keys or set()
        n_skipped = 0

        for query in queries:
            if (strategy.value, query.query_id) in skip:
                n_skipped += 1
                continue
            if generate_answers:
                retrieval_result, generation_result = pipeline.answer(
                    query.query_id, query.text, self._run_top_k, dataset=dataset_name
                )
            else:
                retrieval_result = pipeline.retrieve_only(
                    query.query_id, query.text, self._run_top_k
                )
                generation_result = None

            retrieved_document_ids_by_query[query.query_id] = retrieval_result.top_k_document_ids(
                self._run_top_k
            )

            metrics = {}
            if generation_result is not None:
                answer_score = self._score_answer(
                    dataset_name, query, generation_result.answer_text
                )
                answer_scores.append(answer_score)
                metrics["answer_score"] = answer_score
                if query.query_type.value == "null":
                    abstention_gold.append(True)
                    abstention_predictions.append(generation_result.abstained)

                if judge_llm_client is not None and not generation_result.abstained:
                    context_chunks = retrieval_result.top_k_texts(_JUDGE_CONTEXT_CHUNKS)
                    j_score = llm_judge_score(
                        query.text, context_chunks, generation_result.answer_text, judge_llm_client
                    )
                    judge_scores.append(j_score)
                    metrics["judge_score"] = j_score

            self._evaluation_repository.save_record(
                run_id,
                PipelineRunRecord(
                    query_id=query.query_id,
                    dataset=dataset_name,
                    strategy=strategy,
                    retrieval=retrieval_result,
                    generation=generation_result,
                    metrics=metrics,
                ),
                generation_result,
            )

        # Summary covers only the queries actually processed in this call (skips excluded);
        # when resuming, the authoritative per-pipeline metrics come from aggregating the DB
        # ledger across chunks, not from any single pass's console summary.
        new_queries = [q for q in queries if q.query_id in retrieved_document_ids_by_query]
        retrieval_summary = summarize_retrieval_metrics(
            retrieved_document_ids_by_query,
            {q.query_id: self._judgments_by_query.get(q.query_id, {}) for q in new_queries},
            k=self._run_top_k,
        )
        if n_skipped:
            logger.info(
                "Evaluated %s on %s: %s (skipped %d already-recorded queries)",
                strategy.value, dataset_name, retrieval_summary, n_skipped,
            )
        else:
            logger.info("Evaluated %s on %s: %s", strategy.value, dataset_name, retrieval_summary)

        return DatasetEvaluationReport(
            dataset=dataset_name,
            strategy=strategy,
            retrieval=retrieval_summary,
            mean_answer_score=_safe_mean(answer_scores) if answer_scores else None,
            mean_judge_score=_safe_mean(judge_scores) if judge_scores else None,
            abstention_accuracy=(
                abstention_accuracy(abstention_predictions, abstention_gold)
                if abstention_gold
                else None
            ),
            num_queries=len(queries),
        )

    @staticmethod
    def _score_answer(dataset_name: str, query: Query, predicted_answer: str) -> float:
        """Pick the generation metric that matches `dataset_name`'s answer format."""
        if query.gold_answer is None:
            return 0.0
        if dataset_name == _NUMERIC_ANSWER_DATASET:
            return float(numeric_match(predicted_answer, query.gold_answer))
        if exact_match(predicted_answer, query.gold_answer):
            return 1.0
        return token_f1(predicted_answer, query.gold_answer)

    @staticmethod
    def _group_by_dataset(queries: list[Query]) -> dict[str, list[Query]]:
        grouped: dict[str, list[Query]] = {}
        for query in queries:
            grouped.setdefault(query.source.value, []).append(query)
        return grouped


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def new_run_id() -> str:
    """Generate a fresh, sortable evaluation run id."""
    return f"run-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
