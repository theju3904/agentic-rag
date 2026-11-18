"""Builds the shared `AppContext` once at API startup.

Loading the embedding/reranker models and connecting to Postgres/Qdrant is
expensive, so this happens exactly once in the FastAPI lifespan handler (see
`api/main.py`), not per request.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from agentic_rag.config import Settings
from agentic_rag.domain.retrieval import RetrievalStrategy
from agentic_rag.pipelines.base import RagPipeline
from agentic_rag.pipelines.factory import PipelineFactory
from agentic_rag.storage.postgres.repository import ChunkRepository
from agentic_rag.storage.postgres.session import build_engine, build_session_factory


@dataclass
class AppContext:
    """Every long-lived object the API needs, built once at startup."""

    settings: Settings
    engine: Engine
    session_factory: sessionmaker[Session]
    pipelines: dict[RetrievalStrategy, RagPipeline]


def build_app_context(settings: Settings) -> AppContext:
    """Construct every pipeline and the DB engine/session factory they share."""
    engine = build_engine(settings)
    session_factory = build_session_factory(engine)
    startup_session = session_factory()
    try:
        chunk_repository = ChunkRepository(startup_session)
        factory = PipelineFactory(settings, chunk_repository, startup_session)
        vector_index = factory.build_vector_index()
        all_chunks = chunk_repository.list_all()
        pipelines = factory.build_all_pipelines(all_chunks, vector_index)
    finally:
        startup_session.close()

    return AppContext(
        settings=settings, engine=engine, session_factory=session_factory, pipelines=pipelines
    )
