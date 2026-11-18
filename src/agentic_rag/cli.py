"""Command-line entry point: the composition root for every offline workflow
(build the golden dataset, initialize the DB, index, calibrate, evaluate,
report, serve). Each command wires together already-tested building blocks
from `data/`, `storage/`, `pipelines/`, `evaluation/`, and `agent/`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from agentic_rag.agent.calibration import calibrate_relevance_thresholds
from agentic_rag.config import get_settings
from agentic_rag.data.base_loader import DatasetLoader
from agentic_rag.data.beir_loader import BeirDatasetLoader
from agentic_rag.data.dataset_io import load_dataset_jsonl, save_dataset_jsonl
from agentic_rag.data.golden_dataset_builder import GoldenDatasetBuilder
from agentic_rag.data.multihop_rag_loader import MultiHopRagLoader
from agentic_rag.data.t2_ragbench_loader import T2RagBenchLoader
from agentic_rag.domain.document import DatasetSource
from agentic_rag.domain.retrieval import RetrievalStrategy
from agentic_rag.evaluation.evaluation_runner import EvaluationRunner, new_run_id
from agentic_rag.evaluation.sampling import stratified_sample
from agentic_rag.generation.factory import build_llm_client
from agentic_rag.indexing import IndexingService
from agentic_rag.logging_config import configure_logging
from agentic_rag.pipelines.base import RagPipeline
from agentic_rag.pipelines.factory import PipelineFactory
from agentic_rag.reporting import ReportGenerator, render_report_markdown
from agentic_rag.retrieval.embedding_model import build_embedding_model
from agentic_rag.storage.postgres.repository import (
    ChunkRepository,
    DocumentRepository,
    EvaluationRepository,
    QueryRepository,
)
from agentic_rag.storage.postgres.session import (
    build_engine,
    build_session_factory,
    init_db_schema,
    session_scope,
)
from agentic_rag.storage.vector_index.factory import build_vector_index

logger = logging.getLogger(__name__)
app = typer.Typer(help="Agentic RAG comparison project: build data, index, evaluate, serve.")

_ALL_LOADERS: dict[DatasetSource, DatasetLoader] = {
    DatasetSource.SCIFACT: BeirDatasetLoader("scifact", DatasetSource.SCIFACT),
    DatasetSource.NFCORPUS: BeirDatasetLoader("nfcorpus", DatasetSource.NFCORPUS),
    DatasetSource.FIQA: BeirDatasetLoader("fiqa", DatasetSource.FIQA),
    DatasetSource.MULTIHOP_RAG: MultiHopRagLoader(),
    DatasetSource.T2_RAGBENCH: T2RagBenchLoader(),
}


@app.command("build-dataset")
def build_dataset(
    output_dir: Path = typer.Option(None, help="Defaults to <data_dir>/golden."),
) -> None:
    """Download BEIR + MultiHop-RAG + T2-RAGBench and freeze them into one golden dataset."""
    settings = get_settings()
    configure_logging(settings.log_level)
    target_dir = output_dir or (settings.data_dir / "golden")

    builder = GoldenDatasetBuilder(_ALL_LOADERS, random_seed=settings.random_seed)
    dataset = builder.build()
    save_dataset_jsonl(dataset, target_dir)
    typer.echo(
        f"Wrote golden dataset to {target_dir}: {len(dataset.documents)} docs, "
        f"{len(dataset.queries)} queries, {len(dataset.judgments)} qrels"
    )


@app.command("init-db")
def init_db() -> None:
    """Create every Postgres table if it doesn't already exist (idempotent)."""
    settings = get_settings()
    configure_logging(settings.log_level)
    init_db_schema(build_engine(settings))
    typer.echo("Postgres schema ready.")


@app.command("index")
def index_golden_dataset(golden_dir: Path = typer.Option(None)) -> None:
    """Persist the golden dataset to Postgres and populate the dense vector index."""
    settings = get_settings()
    configure_logging(settings.log_level)
    dataset = load_dataset_jsonl(golden_dir or (settings.data_dir / "golden"))

    engine = build_engine(settings)
    init_db_schema(engine)
    session_factory = build_session_factory(engine)
    embedding_model = build_embedding_model(
        settings.embedding_model_name, settings.remote_embedding_url
    )

    with session_scope(session_factory) as session:
        vector_index = build_vector_index(settings, embedding_model, session)
        indexing_service = IndexingService(
            DocumentRepository(session),
            ChunkRepository(session),
            QueryRepository(session),
            embedding_model,
            vector_index,
        )
        num_chunks = indexing_service.index_dataset(dataset)
    typer.echo(f"Indexed {num_chunks} chunks.")


@app.command("calibrate")
def calibrate(
    golden_dir: Path = typer.Option(None),
    sample_size: int = typer.Option(300, help="Number of queries sampled per score distribution."),
) -> None:
    """Calibrate the agentic pipeline's relevance-grading thresholds on a dev sample."""
    settings = get_settings()
    configure_logging(settings.log_level)
    dataset = load_dataset_jsonl(golden_dir or (settings.data_dir / "golden"))

    thresholds = calibrate_relevance_thresholds(
        dataset, settings.cross_encoder_model_name, sample_size, settings.random_seed
    )
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    thresholds_path = settings.artifacts_dir / "agentic_thresholds.json"
    thresholds_path.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    typer.echo(f"Wrote calibrated thresholds to {thresholds_path}: {thresholds}")


@app.command("evaluate")
def evaluate(
    golden_dir: Path = typer.Option(None),
    top_k: int = typer.Option(
        10, help="Retrieval cutoff used for both ranking metrics and context size."
    ),
    rerank_sample_size: int = typer.Option(
        150,
        help=(
            "Cross-encoder reranking is expensive on CPU (~8-10 pairs/sec, 100-candidate pool "
            "per query), so by default the hybrid_reranked and agentic_corrective pipelines "
            "compute RETRIEVAL metrics on a stratified subsample of this size (by dataset). "
            "Pass 0 to run them over the FULL query set instead (feasible with a GPU reranker, "
            "REMOTE_RERANK_URL). The three cheap pipelines always run over the full set."
        ),
    ),
    generation_sample_size: int = typer.Option(
        100,
        help=(
            "LLM generation is the other CPU bottleneck, so answer-quality metrics "
            "(numeric/exact-match/F1, abstention accuracy) are computed for all five "
            "pipelines on a much smaller stratified subsample of this size."
        ),
    ),
    generation_only: bool = typer.Option(
        False,
        "--generation-only",
        help=(
            "Skip Tiers 1 and 2 (retrieval-only passes) and run only Tier 3 "
            "(retrieve+generate). Use this when only the LLM backend changed and "
            "retrieval scores from a previous run are still valid."
        ),
    ),
    retrieval_only: bool = typer.Option(
        False,
        "--retrieval-only",
        help=(
            "Run all RETRIEVAL tiers (Tier 1 cheap pipelines + Tier 2 reranked/agentic) and "
            "skip the generation Tier 3, so no LLM is used. Combine with --rerank-sample-size 0 "
            "(and a GPU reranker) for a full-set, all-five-pipeline retrieval comparison."
        ),
    ),
    append_to: str = typer.Option(
        "",
        "--append-to",
        help=(
            "Resume an existing run instead of starting a new one. Reuses the given run_id, "
            "skips any (pipeline, query) pair already recorded under it, and processes only "
            "the missing ones. Lets a long full-corpus eval be split across multiple background "
            "chunks without redoing work on restart. The run_id must already exist."
        ),
    ),
    only_strategy: str = typer.Option(
        "",
        "--only-strategy",
        help=(
            "Run only one of the five pipelines (e.g. 'agentic_corrective'). Use when only one "
            "pipeline's code changed and the other four would produce identical results to a "
            "previous run - avoids wasting hours re-running untouched pipelines. The excluded "
            "pipelines are still built (shared infra) but not evaluated."
        ),
    ),
) -> None:
    """Run all five pipelines over the golden dataset and persist results.

    Three-tier evaluation, all under one run id (see README, "Experimental design"):
      1. sparse_bm25/dense/hybrid_rrf: retrieval metrics over every query, no LLM call.
      2. hybrid_reranked/agentic_corrective: retrieval metrics over a stratified subsample
         (cross-encoder reranking is the CPU bottleneck at full scale), no LLM call.
      3. All five pipelines: full retrieve+generate over a much smaller stratified
         subsample, to get answer-quality and abstention metrics without paying for an
         LLM call on every one of the thousands of queries in tiers 1-2.

    Pass --generation-only to skip Tiers 1 and 2 (e.g. when only the LLM backend changed).
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    dataset = load_dataset_jsonl(golden_dir or (settings.data_dir / "golden"))

    engine = build_engine(settings)
    session_factory = build_session_factory(engine)
    run_id = append_to or new_run_id()

    judge_llm_client = build_llm_client(settings)

    with session_scope(session_factory) as session:
        chunk_repository = ChunkRepository(session)
        factory = PipelineFactory(settings, chunk_repository, session)
        vector_index = factory.build_vector_index()
        pipelines = factory.build_all_pipelines(chunk_repository.list_all(), vector_index)

        query_repository = QueryRepository(session)
        evaluation_repository = EvaluationRepository(session)
        runner = EvaluationRunner(
            evaluation_repository,
            query_repository.judgments_by_query(),
            run_top_k=top_k,
            judge_llm_client=judge_llm_client,
        )

        skip_keys: set[tuple[str, str]] = set()
        if append_to:
            if not evaluation_repository.run_exists(append_to):
                raise typer.BadParameter(f"--append-to run_id '{append_to}' does not exist.")
            skip_keys = evaluation_repository.completed_keys(append_to)
            typer.echo(
                f"Resuming run {append_to}: {len(skip_keys)} (pipeline, query) pairs "
                f"already recorded will be skipped."
            )

        cheap_strategies = (
            RetrievalStrategy.SPARSE_BM25,
            RetrievalStrategy.DENSE,
            RetrievalStrategy.HYBRID_RRF,
        )
        expensive_strategies = (
            RetrievalStrategy.HYBRID_RERANKED,
            RetrievalStrategy.AGENTIC_CORRECTIVE,
        )
        if only_strategy:
            try:
                only = RetrievalStrategy(only_strategy)
            except ValueError:
                allowed = ", ".join(s.value for s in RetrievalStrategy)
                raise typer.BadParameter(
                    f"--only-strategy '{only_strategy}' is not a valid strategy. "
                    f"Choose one of: {allowed}"
                ) from None
            pipelines = {only: pipelines[only]}
            typer.echo(f"--only-strategy: evaluating ONLY {only.value}")
        else:
            # Default evaluation runs only the five top-level pipelines. The
            # agentic sub-strategy LEARNED_ROUTER shares its retriever with the
            # agentic pipeline's learned_router mode, so including it by default
            # would repeat hours of work and write identical rows under two
            # names. Pass `--only-strategy learned_router` to record it
            # separately (e.g. to produce rows for `train-router` to reuse).
            pipelines = {
                s: p for s, p in pipelines.items() if s.is_top_level_pipeline
            }

        cheap_pipelines: dict[RetrievalStrategy, RagPipeline] = {
            strategy: pipeline
            for strategy, pipeline in pipelines.items()
            if strategy in cheap_strategies
        }
        expensive_pipelines: dict[RetrievalStrategy, RagPipeline] = {
            strategy: pipeline
            for strategy, pipeline in pipelines.items()
            if strategy in expensive_strategies
        }
        rerank_subsample = (
            dataset.queries
            if rerank_sample_size == 0
            else stratified_sample(
                dataset.queries,
                max_total=rerank_sample_size,
                group_key=lambda q: q.source.value,
                random_seed=settings.random_seed,
            )
        )
        generation_subsample = stratified_sample(
            dataset.queries,
            max_total=generation_sample_size,
            group_key=lambda q: q.source.value,
            random_seed=settings.random_seed,
        )

        if not append_to:
            runner.start(run_id, config_snapshot=_redacted_config(settings))
        reports: list = []

        if generation_only and retrieval_only:
            raise typer.BadParameter(
                "--generation-only and --retrieval-only are mutually exclusive."
            )

        if retrieval_only:
            # All five pipelines on ONE shared query set (no LLM), for a clean,
            # apples-to-apples retrieval comparison including reranked + agentic.
            # rerank_sample_size == 0 means the full set; feasible for the reranked
            # pipelines only with a GPU reranker (REMOTE_RERANK_URL).
            eval_set = dataset.queries if rerank_sample_size == 0 else rerank_subsample
            n_pipelines = len(cheap_pipelines) + len(expensive_pipelines)
            typer.echo(
                f"Retrieval-only: {n_pipelines} pipeline(s) on {len(eval_set)} "
                "shared queries (no LLM)"
            )
            reports += runner.run_pass(
                run_id, cheap_pipelines, eval_set, generate_answers=False, skip_keys=skip_keys,
            )
            reports += runner.run_pass(
                run_id, expensive_pipelines, eval_set, generate_answers=False, skip_keys=skip_keys,
            )
        else:
            if not generation_only:
                typer.echo(f"Tier 1 (retrieval-only, full set): {len(dataset.queries)} queries")
                reports += runner.run_pass(
                    run_id, cheap_pipelines, dataset.queries, generate_answers=False,
                    skip_keys=skip_keys,
                )
                typer.echo(
                    f"Tier 2 (retrieval-only, rerank subsample): {len(rerank_subsample)} queries"
                )
                reports += runner.run_pass(
                    run_id, expensive_pipelines, rerank_subsample, generate_answers=False,
                    skip_keys=skip_keys,
                )
            else:
                typer.echo("Skipping Tiers 1 & 2 (--generation-only).")

            typer.echo(
                f"Tier 3 (retrieve+generate, all 5 pipelines): {len(generation_subsample)} queries"
            )
            reports += runner.run_pass(
                run_id, pipelines, generation_subsample, generate_answers=True,
                skip_keys=skip_keys,
            )
        runner.finish(run_id)

    typer.echo(f"Run {run_id} complete.")
    for report in reports:
        answer_score_display = (
            f"{report.mean_answer_score:.3f}" if report.mean_answer_score is not None else "n/a"
        )
        judge_score_display = (
            f"{report.mean_judge_score:.3f}" if report.mean_judge_score is not None else "n/a"
        )
        typer.echo(
            f"  {report.dataset:15s} {report.strategy.value:20s} "
            f"nDCG@{top_k}={report.retrieval.ndcg_at_k:.3f} "
            f"answer_score={answer_score_display} "
            f"judge_score={judge_score_display}"
        )


@app.command("report")
def report(
    run_id: str,
    output: Path = typer.Option(Path("report/results.md")),
) -> None:
    """Render the final Markdown results tables for a previously completed `run_id`."""
    settings = get_settings()
    configure_logging(settings.log_level)
    engine = build_engine(settings)
    session_factory = build_session_factory(engine)

    with session_scope(session_factory) as session:
        generator = ReportGenerator(EvaluationRepository(session), QueryRepository(session))
        retrieval_table = generator.build_retrieval_table(run_id)
        comparisons = generator.compare_agentic_vs_baselines(
            run_id,
            baseline_strategies=["sparse_bm25", "dense", "hybrid_rrf", "hybrid_reranked"],
        )
        if settings.llm_backend == "openai":
            model_name = settings.openai_model
        elif settings.llm_backend == "gemma":
            model_name = settings.gemma_model
        else:
            model_name = settings.local_generation_model
        cost_latency_table = generator.build_cost_latency_table(run_id, model_name)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_report_markdown(run_id, retrieval_table, comparisons, cost_latency_table),
        encoding="utf-8",
    )
    typer.echo(f"Wrote report to {output}")


@app.command("train-router")
def train_router(
    run_ids: list[str] = typer.Argument(
        ...,
        help=(
            "One or more completed pipeline-run ids whose records cover BM25, Dense, "
            "hybrid+rerank, and agentic. Required for training the learned router."
        ),
    ),
    delta: float = typer.Option(
        0.10,
        help=(
            "Tie-aware target margin: a non-P5 strategy is only labelled as the "
            "winner when it beats P5 by at least this nDCG; otherwise the label "
            "stays agentic_corrective so the router defaults to P5."
        ),
    ),
    top_k: int = typer.Option(10, help="nDCG cutoff used for the training target."),
) -> None:
    """Train the learned-router (`LEARNED_ROUTER`) XGBoost classifier from a ledger and save it."""
    from agentic_rag.agent.learned_router import artifact_dir_from_settings
    from agentic_rag.agent.learned_router_training import train_learned_router

    settings = get_settings()
    configure_logging(settings.log_level)
    engine = build_engine(settings)
    out_dir = artifact_dir_from_settings(settings)
    summary = train_learned_router(engine, run_ids, out_dir, top_k=top_k, delta=delta)
    typer.echo(f"Trained learned router from {summary['n_training_queries']} queries "
               f"(train acc {summary['train_accuracy']:.4f}); wrote artifacts to {out_dir}")


@app.command("serve")
def serve(
    host: str = typer.Option(None),
    port: int = typer.Option(None),
) -> None:
    """Run the FastAPI app with uvicorn."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "agentic_rag.api.main:create_app",
        factory=True,
        host=host or settings.api_host,
        port=port or settings.api_port,
    )


def _redacted_config(settings) -> dict:
    """Build a config snapshot safe to persist: never store API keys/passwords at rest."""
    return {
        "llm_backend": settings.llm_backend,
        "openai_model": settings.openai_model if settings.llm_backend == "openai" else None,
        "local_generation_model": settings.local_generation_model,
        "embedding_model_name": settings.embedding_model_name,
        "cross_encoder_model_name": settings.cross_encoder_model_name,
        "vector_backend": settings.vector_backend,
        "random_seed": settings.random_seed,
    }


if __name__ == "__main__":
    app()
