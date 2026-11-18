"""Engine/session factory for the Postgres system of record."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from agentic_rag.config import Settings
from agentic_rag.storage.postgres.models import Base


def build_engine(settings: Settings) -> Engine:
    """Create a SQLAlchemy engine for the configured Postgres instance."""
    return create_engine(settings.postgres_dsn, pool_pre_ping=True, future=True)


def init_db_schema(engine: Engine) -> None:
    """Create every ORM-mapped table if it doesn't already exist (idempotent).

    A full migration tool (Alembic) would be the production choice; this
    project uses one `create_all` call since the schema has no versioned
    history to migrate between (see README, "What's simplified").
    """
    Base.metadata.create_all(engine)


def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create a session factory bound to `engine`."""
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@contextmanager
def session_scope(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    """Provide a transactional session that commits on success, rolls back on error."""
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
