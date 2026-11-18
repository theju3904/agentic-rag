"""Generates the final Markdown results tables from a persisted evaluation run.

Reads directly from the Postgres ledger (`PipelineRunRecordRow`) rather than
re-running any pipeline, so the report can be regenerated cheaply at any time.
"""

from __future__ import annotations

from collections import defaultdict

from agentic_rag.evaluation.cost_tracking import summarize_cost
from agentic_rag.evaluation.retrieval_metrics import summarize_retrieval_metrics
from agentic_rag.evaluation.statistics import paired_bootstrap_comparison
from agentic_rag.storage.postgres.models import PipelineRunRecordRow
from agentic_rag.storage.postgres.repository import EvaluationRepository, QueryRepository


class ReportGenerator:
    """Recomputes retrieval metrics and pipeline comparisons from a run's ledger rows."""

    def __init__(
        self, evaluation_repository: EvaluationRepository, query_repository: QueryRepository
    ) -> None:
        self._evaluation_repository = evaluation_repository
        self._query_repository = query_repository

    def build_retrieval_table(self, run_id: str, k: int = 10) -> list[dict]:
        """Return one row per (dataset, strategy) with Recall/nDCG/MRR/MAP@k and answer score."""
        rows = self._evaluation_repository.fetch_run_records(run_id)
        judgments_by_query = self._query_repository.judgments_by_query()

        rows_by_dataset_strategy: dict[tuple[str, str], list[PipelineRunRecordRow]] = defaultdict(
            list
        )
        for row in rows:
            rows_by_dataset_strategy[(row.dataset, row.strategy)].append(row)

        table = []
        for (dataset, strategy), group_rows in rows_by_dataset_strategy.items():
            retrieved_by_query = {row.query_id: row.retrieved_document_ids for row in group_rows}
            judgments_subset = {
                row.query_id: judgments_by_query.get(row.query_id, {}) for row in group_rows
            }
            summary = summarize_retrieval_metrics(retrieved_by_query, judgments_subset, k)
            # Retrieval-only rows (see EvaluationRunner's `generate_answers=False`
            # pass) have no "answer_score" key; they must not be averaged in as 0.
            answer_scores = [
                row.metrics["answer_score"] for row in group_rows if "answer_score" in row.metrics
            ]
            judge_scores = [
                row.metrics["judge_score"] for row in group_rows if "judge_score" in row.metrics
            ]
            table.append(
                {
                    "dataset": dataset,
                    "strategy": strategy,
                    "recall_at_k": summary.recall_at_k,
                    "ndcg_at_k": summary.ndcg_at_k,
                    "mrr_at_k": summary.mrr_at_k,
                    "map_at_k": summary.map_at_k,
                    "mean_answer_score": sum(answer_scores) / len(answer_scores)
                    if answer_scores
                    else None,
                    "mean_judge_score": sum(judge_scores) / len(judge_scores)
                    if judge_scores
                    else None,
                    "num_generated": len(answer_scores),
                    "num_queries": summary.num_queries,
                }
            )
        return sorted(table, key=lambda row: (row["dataset"], row["strategy"]))

    def build_cost_latency_table(self, run_id: str, model_name: str) -> list[dict]:
        """Return one row per (dataset, strategy) with mean latency and generation cost."""
        rows = self._evaluation_repository.fetch_run_records(run_id)
        rows_by_dataset_strategy: dict[tuple[str, str], list[PipelineRunRecordRow]] = defaultdict(
            list
        )
        for row in rows:
            rows_by_dataset_strategy[(row.dataset, row.strategy)].append(row)

        table = []
        for (dataset, strategy), group_rows in rows_by_dataset_strategy.items():
            generation_rows = [
                row for row in group_rows if row.generation_latency_seconds is not None
            ]
            cost_summary = summarize_cost(
                retrieval_latencies=[row.retrieval_latency_seconds for row in group_rows],
                generation_latencies=[
                    row.generation_latency_seconds
                    for row in generation_rows
                    if row.generation_latency_seconds is not None
                ],
                prompt_token_counts=[row.prompt_tokens or 0 for row in generation_rows],
                completion_token_counts=[row.completion_tokens or 0 for row in generation_rows],
                model_name=model_name,
            )
            table.append(
                {
                    "dataset": dataset,
                    "strategy": strategy,
                    "mean_retrieval_latency_s": cost_summary.mean_retrieval_latency_seconds,
                    "mean_generation_latency_s": cost_summary.mean_generation_latency_seconds,
                    "total_prompt_tokens": cost_summary.total_prompt_tokens,
                    "total_completion_tokens": cost_summary.total_completion_tokens,
                    "estimated_cost_usd": cost_summary.estimated_cost_usd,
                }
            )
        return sorted(table, key=lambda row: (row["dataset"], row["strategy"]))

    def compare_agentic_vs_baselines(
        self, run_id: str, baseline_strategies: list[str], metric_key: str = "answer_score"
    ) -> list[dict]:
        """Paired-bootstrap-compare the agentic pipeline against each baseline, per dataset."""
        rows = self._evaluation_repository.fetch_run_records(run_id)
        scores_by_dataset_strategy: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
        for row in rows:
            # Retrieval-only rows (no generation pass) have no `metric_key` in
            # `metrics` at all; they must be skipped, not defaulted to 0.0.
            if metric_key in row.metrics:
                scores_by_dataset_strategy[(row.dataset, row.strategy)][row.query_id] = row.metrics[
                    metric_key
                ]

        datasets = sorted({dataset for dataset, _ in scores_by_dataset_strategy})
        comparisons = []
        for dataset in datasets:
            agentic_scores_by_query = scores_by_dataset_strategy.get(
                (dataset, "agentic_corrective")
            )
            if not agentic_scores_by_query:
                continue
            for baseline in baseline_strategies:
                comparison = self._compare_one_baseline(
                    scores_by_dataset_strategy, dataset, baseline, agentic_scores_by_query
                )
                if comparison is not None:
                    comparisons.append(comparison)
        return comparisons

    @staticmethod
    def _compare_one_baseline(
        scores_by_dataset_strategy: dict[tuple[str, str], dict[str, float]],
        dataset: str,
        baseline: str,
        agentic_scores_by_query: dict[str, float],
    ) -> dict | None:
        baseline_scores_by_query = scores_by_dataset_strategy.get((dataset, baseline))
        if not baseline_scores_by_query:
            return None
        shared_query_ids = sorted(set(agentic_scores_by_query) & set(baseline_scores_by_query))
        if not shared_query_ids:
            return None

        agentic_scores = [agentic_scores_by_query[qid] for qid in shared_query_ids]
        baseline_scores = [baseline_scores_by_query[qid] for qid in shared_query_ids]
        result = paired_bootstrap_comparison(agentic_scores, baseline_scores)
        return {
            "dataset": dataset,
            "baseline": baseline,
            "n": len(shared_query_ids),
            "mean_difference": result.mean_difference,
            "ci_low": result.confidence_interval_low,
            "ci_high": result.confidence_interval_high,
            "p_value": result.p_value,
        }


def render_markdown_table(rows: list[dict]) -> str:
    """Render a list of flat dicts (all with the same keys) as a Markdown table."""
    if not rows:
        return "_(no rows)_\n"
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row[header]) for header in headers) + " |")
    return "\n".join(lines) + "\n"


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_report_markdown(
    run_id: str,
    retrieval_table: list[dict],
    comparisons: list[dict],
    cost_latency_table: list[dict] | None = None,
) -> str:
    """Render the full Markdown report for one evaluation run."""
    sections = [
        f"# Evaluation report: `{run_id}`\n",
        "## Retrieval + answer-quality metrics, per (dataset, strategy)\n",
        render_markdown_table(retrieval_table),
        "\n## Agentic pipeline vs. each baseline (paired bootstrap, answer_score)\n",
        render_markdown_table(comparisons),
    ]
    if cost_latency_table is not None:
        sections += [
            "\n## Cost & latency, per (dataset, strategy)\n",
            render_markdown_table(cost_latency_table),
        ]
    return "\n".join(sections)
