"""Engine and session management. ``DATABASE_URL`` selects SQLite (default) or PostgreSQL."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from trading_mcp.config import get_settings
from trading_mcp.db.models import Base

_engine: Engine | None = None
_factory: sessionmaker[Session] | None = None
_lock = threading.Lock()


def normalize_url(url: str) -> str:
    """Accept Heroku/Render style ``postgres://`` URLs and use the psycopg 3 driver."""
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


def get_engine() -> Engine:
    global _engine, _factory
    with _lock:
        if _engine is None:
            url = normalize_url(get_settings().database_url)
            kwargs: dict[str, object] = {"pool_pre_ping": True}
            if url.startswith("sqlite"):
                db_path = url.split("///", 1)[-1]
                if db_path and db_path != ":memory:":
                    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
                kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
            engine = create_engine(url, **kwargs)
            if url.startswith("sqlite"):
                event.listen(engine, "connect", _sqlite_pragmas)
            Base.metadata.create_all(engine)  # idempotent; use Alembic for schema migrations later
            _engine, _factory = engine, sessionmaker(engine, expire_on_commit=False)
        return _engine


def _sqlite_pragmas(dbapi_conn, _record) -> None:  # noqa: ANN001
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")  # concurrent readers while the server writes
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    get_engine()
    assert _factory is not None
    session = _factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine() -> None:
    """Dispose the engine (tests / settings changes)."""
    global _engine, _factory
    with _lock:
        if _engine is not None:
            _engine.dispose()
        _engine, _factory = None, None
