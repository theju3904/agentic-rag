"""Centralized, environment-driven application configuration.

Every tunable that affects reproducibility (model names, seeds, connection
strings) lives on `Settings` so the whole configuration can be logged and
serialized alongside evaluation results for later auditing.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# pydantic-settings reads .env into Settings fields but never into os.environ,
# so load it here for the os.getenv-based port lookups below (real env vars win).
load_dotenv(PROJECT_ROOT / ".env")


def _require_env(name: str) -> str:
    """Return an environment variable, failing loudly if .env doesn't define it."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Define it in .env (see .env.example).")
    return value


class Settings(BaseSettings):
    """Application settings, loaded from environment variables / a `.env` file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Postgres: source-of-truth store for documents, chunks, qrels, eval runs ---
    postgres_host: str = "localhost"
    postgres_port: int = Field(default_factory=lambda: int(_require_env("POSTGRES_PORT")))
    postgres_db: str = "agentic_rag"
    postgres_user: str = "agentic_rag"
    postgres_password: str = "change-me"

    # --- Qdrant: ANN index backing the dense/hybrid/agentic pipelines ---
    qdrant_url: str = Field(
        default_factory=lambda: f"http://localhost:{_require_env('QDRANT_HTTP_PORT')}"
    )
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "agentic_rag_chunks"

    # --- Which VectorIndex implementation to use: "qdrant" | "pgvector" ---
    vector_backend: str = "qdrant"

    # --- Pluggable LLM backend: "gemma" | "openai" | "local" | "null" ---
    # "gemma" (default): self-hosted Gemma via Ollama /api/chat; auto-falls back
    # to gpt-4o-mini when OPENAI_API_KEY is set and the Gemma endpoint errors.
    llm_backend: str = "gemma"
    gemma_base_url: str = Field(
        default_factory=lambda: f"http://{_require_env('GPU_URL')}:{_require_env('GEMMA_PORT')}"
    )
    gemma_model: str = "gemma4:E4B"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    local_generation_model: str = "google/flan-t5-base"

    # --- Retrieval models ---
    # Set REMOTE_EMBEDDING_URL to offload embedding to a GPU server (HTTP API).
    # When set, embedding_model_name is sent as the "model" field in each request.
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    remote_embedding_url: str | None = None
    cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Set REMOTE_RERANK_URL to offload cross-encoder reranking to a GPU server.
    # The server must run the SAME cross_encoder_model_name in fp32 so the raw
    # scores stay identical to the CPU path (the agentic thresholds are calibrated
    # against those exact scores).
    remote_rerank_url: str | None = None

    # Agentic pipeline mode selector. All three are agentic in the project's sense:
    # each makes a per-query decision about which retrieval path to take. They differ
    # only in the decision function.
    #
    #   "learned_router" (RECOMMENDED) - XGBoost classifier picks per query among BM25 /
    #     Dense / Hybrid+Rerank / heuristic_router after running them all; uses 12 query-text
    #     features + 4 retrieval-overlap features. Full-set winner: mean nDCG@10 = 0.521
    #     vs 0.483 heuristic_router, wins every dataset against every classic baseline.
    #     Requires a trained artifact at `artifacts/router/learned_router.xgb.json`
    #     (produced by `agentic-rag train-router`); falls back to "heuristic_router"
    #     if the artifact is missing.
    #   "heuristic_router" (DEFAULT when no artifact exists) - classifies each query by
    #     text features and routes to BM25 (no rerank) / Hybrid+Rerank. Zero-training
    #     baseline; mean nDCG@10 = 0.483.
    #   "per_doc" - CRAG-style hybrid+rerank+per-document grading with query rewrite.
    #     Correct per spec; grader filter is nearly inert under calibrated thresholds.
    # Set AGENTIC_MODE to switch at run time without changing code.
    agentic_mode: str = "learned_router"

    # --- Reproducibility ---
    random_seed: int = 42

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = Field(default_factory=lambda: int(_require_env("API_PORT")))
    log_level: str = "INFO"

    # --- Paths ---
    data_dir: Path = PROJECT_ROOT / "data"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"

    @property
    def postgres_dsn(self) -> str:
        """Build a psycopg3 SQLAlchemy connection string from discrete fields."""
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide `Settings` singleton, constructed on first call."""
    return Settings()
