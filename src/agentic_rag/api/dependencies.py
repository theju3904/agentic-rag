"""FastAPI dependency providers: per-request sessions and the shared app context."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Request
from sqlalchemy.orm import Session

from agentic_rag.api.context import AppContext


def get_app_context(request: Request) -> AppContext:
    """Return the `AppContext` built once at startup (see `api/main.py`)."""
    return request.app.state.context


def get_session(request: Request) -> Iterator[Session]:
    """Yield a fresh Postgres session for the duration of one request."""
    context: AppContext = request.app.state.context
    session = context.session_factory()
    try:
        yield session
    finally:
        session.close()
