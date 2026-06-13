"""Engine / session factory + schema bootstrap (doc 04 §2.1).

Access is via **SQLAlchemy** with a thin spatial helper layer (doc 04 §2.1). The
primary engine is **PostgreSQL + PostGIS** (doc 04 §2.1 *the choice*); an embedded
**SQLite** file/in-memory build is the documented lightweight fallback used by
tests (no service required). This module hides the URL choice behind one factory:

- ``default_sqlite_url()`` — the local/test fallback (in-memory by default).
- ``make_engine(url)`` — builds an Engine, enabling SQLite FK enforcement (so the
  doc 04 §2.4 ``ON DELETE CASCADE`` constraints actually fire on SQLite, which
  has foreign keys *off* by default).
- ``session_factory(engine)`` / ``create_all(engine)`` — standard bootstrap.
- ``is_postgis(engine)`` — capability flag gating the PostGIS GiST path (the bbox
  helper in ``geosim.catalog.spatial`` keys on it; doc 04 §2.5).

Tests use ``create_all`` against SQLite in-memory; Alembic migrations
(``geosim.catalog.migrations``) are the production schema path.
"""

from __future__ import annotations

from sqlalchemy import Engine, event
from sqlalchemy import create_engine as _sa_create_engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

__all__ = [
    "default_sqlite_url",
    "make_engine",
    "session_factory",
    "create_all",
    "drop_all",
    "is_postgis",
    "is_sqlite",
]


def default_sqlite_url(path: str | None = None) -> str:
    """SQLite URL for the lightweight fallback (doc 04 §2.1).

    With ``path`` None this is a shared in-memory DB suitable for tests; pass a
    filesystem path for an embedded single-file demo store.
    """
    if path is None:
        return "sqlite+pysqlite:///:memory:"
    return f"sqlite+pysqlite:///{path}"


def make_engine(url: str | None = None, *, echo: bool = False, **kwargs) -> Engine:
    """Build an Engine for ``url`` (defaults to the SQLite fallback, doc 04 §2.1).

    For SQLite we install a ``PRAGMA foreign_keys=ON`` hook on every connection so
    the ``ON DELETE CASCADE`` foreign keys (doc 04 §2.4) are enforced — SQLite
    leaves them off otherwise — and use a ``StaticPool`` for ``:memory:`` so an
    in-memory DB survives across sessions within one Engine (needed for tests).
    """
    url = url or default_sqlite_url()
    if url.startswith("sqlite"):
        if ":memory:" in url:
            from sqlalchemy.pool import StaticPool

            kwargs.setdefault("poolclass", StaticPool)
            kwargs.setdefault("connect_args", {"check_same_thread": False})
        engine = _sa_create_engine(url, echo=echo, **kwargs)

        @event.listens_for(engine, "connect")
        def _enable_sqlite_fk(dbapi_conn, _record):  # pragma: no cover - trivial hook
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

        return engine
    return _sa_create_engine(url, echo=echo, **kwargs)


def session_factory(engine: Engine) -> sessionmaker[Session]:
    """A configured ``sessionmaker`` bound to ``engine``."""
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def create_all(engine: Engine) -> None:
    """Create every catalog table (doc 04 §2.4). Tests use this; prod uses Alembic."""
    Base.metadata.create_all(engine)


def drop_all(engine: Engine) -> None:
    """Drop every catalog table (test teardown convenience)."""
    Base.metadata.drop_all(engine)


def is_sqlite(engine: Engine) -> bool:
    """True if ``engine`` targets SQLite (the portable bbox path, doc 04 §2.5)."""
    return engine.dialect.name == "sqlite"


def is_postgis(engine: Engine) -> bool:
    """Capability flag: True if the PostGIS GiST bbox path is available (doc 04 §2.5).

    Confined here so PostGIS-specific SQL stays behind a flag (doc 04 §2.1). On a
    plain PostgreSQL connection we probe for the ``postgis`` extension; any other
    dialect (SQLite fallback) reports False and uses the portable range query.
    """
    if engine.dialect.name != "postgresql":
        return False
    try:
        with engine.connect() as conn:
            from sqlalchemy import text

            row = conn.execute(
                text("SELECT 1 FROM pg_extension WHERE extname = 'postgis'")
            ).first()
            return row is not None
    except Exception:  # pragma: no cover - depends on a live PG server
        return False
