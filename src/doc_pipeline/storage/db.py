"""Database engine, session management, and master supplier seeding."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, event, select, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from doc_pipeline.config import Settings, get_settings
from doc_pipeline.storage.models import AuditLogEntry, Base, SupplierMaster
from doc_pipeline.synthetic.catalog import MASTER_SUPPLIERS

_engines: dict[str, Engine] = {}


def get_engine(database_url: str | None = None) -> Engine:
    url = database_url or get_settings().database_url
    if url not in _engines:
        if url.startswith("sqlite"):
            connect_args = {"check_same_thread": False, "timeout": 30}
            if ":memory:" in url:
                _engines[url] = create_engine(
                    url,
                    connect_args=connect_args,
                    poolclass=StaticPool,
                )
            else:
                eng = create_engine(url, connect_args=connect_args)

                @event.listens_for(eng, "connect")
                def _set_sqlite_wal(dbapi_conn, _conn_record) -> None:
                    cur = dbapi_conn.cursor()
                    cur.execute("PRAGMA journal_mode=WAL")
                    cur.execute("PRAGMA synchronous=NORMAL")
                    cur.execute("PRAGMA busy_timeout=30000")
                    cur.close()

                _engines[url] = eng
        else:
            _engines[url] = create_engine(url, pool_pre_ping=True, pool_size=10, max_overflow=20)
    return _engines[url]


def dispose_engine(database_url: str) -> None:
    """Close pooled connections and remove an engine (required before deleting SQLite temp files on Windows)."""
    eng = _engines.pop(database_url, None)
    if eng is not None:
        eng.dispose()


def make_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    engine = get_engine(database_url)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


@contextmanager
def session_scope(settings: Settings | None = None) -> Generator[Session, None, None]:
    cfg = settings or get_settings()
    factory = make_session_factory(cfg.database_url)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def seed_master_suppliers(session: Session) -> int:
    """Idempotently populate the master suppliers directory."""
    inserted = 0
    for sup in MASTER_SUPPLIERS:
        existing = session.get(SupplierMaster, sup["id"])
        if existing is None:
            session.add(
                SupplierMaster(
                    id=sup["id"],
                    name=sup["name"],
                    aliases=sup["aliases"],
                    tax_id=sup["tax_id"],
                    address=sup["address"],
                    default_currency=sup["default_currency"],
                    country=sup["country"],
                    active=sup["active"],
                )
            )
            inserted += 1
    session.flush()
    return inserted


def init_db(database_url: str | None = None) -> None:
    """Create schema tables if missing and seed master suppliers (serialized on Postgres)."""
    engine = get_engine(database_url)
    is_pg = engine.dialect.name == "postgresql"
    with engine.connect() as lock_conn:
        if is_pg:
            lock_conn.execute(text("SELECT pg_advisory_lock(74829104)"))
            lock_conn.commit()
        try:
            Base.metadata.create_all(bind=engine)
            factory = sessionmaker(
                bind=engine, autoflush=False, autocommit=False, expire_on_commit=False
            )
            with factory() as session:
                existing_count = len(session.scalars(select(SupplierMaster.id)).all())
                if existing_count < len(MASTER_SUPPLIERS):
                    seed_master_suppliers(session)
                    session.commit()
        finally:
            if is_pg:
                lock_conn.execute(text("SELECT pg_advisory_unlock(74829104)"))
                lock_conn.commit()


def append_audit_log(
    session: Session,
    *,
    event_type: str,
    summary: str,
    document_id: str | None = None,
    actor: str = "system",
    payload: dict | None = None,
) -> AuditLogEntry:
    """Record an immutable audit log entry inside the active transaction."""
    entry = AuditLogEntry(
        document_id=document_id,
        event_type=event_type,
        actor=actor,
        summary=summary,
        payload=payload or {},
    )
    session.add(entry)
    session.flush()
    return entry
