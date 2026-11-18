"""FastAPI application factory.

Builds every pipeline once at startup (loading embedding/reranker models and
connecting to Postgres/Qdrant is expensive) and stores them on `app.state`,
rather than reconstructing them on every request.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from agentic_rag.api.context import build_app_context
from agentic_rag.api.demo_page import DEMO_HTML
from agentic_rag.api.routers.documents_router import chunks_router, documents_router
from agentic_rag.api.routers.evaluation_router import router as evaluation_router
from agentic_rag.api.routers.pipelines_router import router as pipelines_router
from agentic_rag.api.routers.query_router import router as query_router
from agentic_rag.config import get_settings
from agentic_rag.logging_config import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("Building application context (models, Postgres, Qdrant)...")
    app.state.context = build_app_context(settings)
    logger.info("Application context ready; %d pipelines loaded.", len(app.state.context.pipelines))
    yield
    app.state.context.engine.dispose()


def create_app() -> FastAPI:
    """Application factory used by both `uvicorn --factory` and the test suite."""
    app = FastAPI(
        title="Agentic RAG Comparison API",
        description="Inspect chunks/documents and query any of the five RAG pipelines.",
        version="0.1.0",
        lifespan=_lifespan,
    )
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def demo_ui() -> HTMLResponse:
        return HTMLResponse(DEMO_HTML)

    app.include_router(documents_router)
    app.include_router(chunks_router)
    app.include_router(query_router)
    app.include_router(pipelines_router)
    app.include_router(evaluation_router)

    @app.get("/health", tags=["meta"])
    def health() -> dict:
        return {"status": "ok"}

    return app
