"""Shared API-test fixtures.

The API tests run against the seeded Postgres (loaded by Phase 3's
``test_query_plans.py`` conftest). If lap_times is empty here — e.g.
because the ETL tests just truncated everything — we re-seed.

The async engine in ``app.db.session`` caches connections across the
process; ``TestClient`` runs each request on a fresh event-loop portal,
which breaks pooled asyncpg connections. We work around this by
overriding the ``get_db`` dependency per-test with a fresh ``NullPool``
engine that opens-and-closes one connection per request.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.db.session import get_db
from app.main import app
from tests.test_query_plans import SEED_SQL_PATH  # reuse the same seed file

_SEQUENCE_BUMP = """
SELECT setval(pg_get_serial_sequence(table_name, 'id'),
              COALESCE((SELECT MAX(id) FROM only_table), 1), true)
FROM (VALUES
  ('seasons'), ('circuits'), ('events'), ('sessions'),
  ('teams'), ('drivers'), ('race_results'), ('lap_times')
) AS t(table_name),
LATERAL (SELECT 0 AS only_table) z
"""


@pytest.fixture(scope="package", autouse=True)
def _ensure_seeded() -> None:
    engine = sa.create_engine(settings.DATABASE_URL_SYNC, future=True)
    with engine.connect() as conn:
        n = conn.scalar(sa.text("SELECT COUNT(*) FROM lap_times")) or 0
    if n < 100_000:
        sql = SEED_SQL_PATH.read_text()
        raw = engine.raw_connection()
        try:
            raw.cursor().execute(sql)
            raw.commit()
        finally:
            raw.close()

    # The seed inserts rows with explicit IDs (1..N) via plain INSERTs, which
    # leaves the SERIAL sequences at 1. Any subsequent fixture INSERT then
    # collides. Bump each sequence past its current MAX(id).
    with engine.begin() as conn:
        for table in (
            "seasons", "circuits", "events", "sessions",
            "teams", "drivers", "race_results", "lap_times",
        ):
            conn.execute(sa.text(
                f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), "
                f"COALESCE((SELECT MAX(id) FROM {table}), 1), true)"
            ))


@pytest.fixture()
def client() -> Iterator[TestClient]:
    """Per-test fresh NullPool engine, used by BOTH the get_db dependency
    AND the UserSessionMiddleware (which holds its own session). Without
    the middleware swap, the cross-event-loop guard in asyncpg blows up
    when TestClient creates a new portal per request.
    """
    test_engine = create_async_engine(settings.DATABASE_URL, poolclass=NullPool)
    TestSessionLocal = async_sessionmaker(
        bind=test_engine, expire_on_commit=False, autoflush=False
    )

    async def _override_get_db() -> AsyncIterator:
        async with TestSessionLocal() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise

    import app.core.sessions as _ms_mod
    import app.db.session as _sess_mod
    _orig_engine = _sess_mod.engine
    _orig_factory = _sess_mod.AsyncSessionLocal
    _sess_mod.engine = test_engine
    _sess_mod.AsyncSessionLocal = TestSessionLocal
    _ms_mod.AsyncSessionLocal = TestSessionLocal

    app.dependency_overrides[get_db] = _override_get_db
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_db, None)
        _sess_mod.engine = _orig_engine
        _sess_mod.AsyncSessionLocal = _orig_factory
        _ms_mod.AsyncSessionLocal = _orig_factory
