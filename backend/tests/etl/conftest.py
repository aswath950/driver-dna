"""Shared ETL fixtures.

Each test gets a fully-truncated database so row counts are deterministic.
Uses the same Postgres instance as the rest of the suite (settings.DATABASE_URL_SYNC).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.core.config import settings

_TABLES_IN_FK_ORDER = [
    "driver_stats",
    "lap_times",
    "race_results",
    "session_drivers",
    "sessions",
    "events",
    "drivers",
    "teams",
    "circuits",
    "seasons",
]


@pytest.fixture(scope="session")
def engine() -> sa.Engine:
    eng = sa.create_engine(settings.DATABASE_URL_SYNC, future=True)
    try:
        with eng.connect():
            pass
    except sa.exc.OperationalError as e:
        pytest.skip(f"Postgres not reachable: {e}")
    return eng


@pytest.fixture()
def clean_db(engine: sa.Engine) -> Iterator[sa.Engine]:
    """TRUNCATE all 10 tables before AND after each ETL test so they're
    isolated from each other and from the seed data Phase 3 loaded."""
    stmt = sa.text(
        "TRUNCATE " + ", ".join(_TABLES_IN_FK_ORDER) + " RESTART IDENTITY CASCADE"
    )
    with engine.begin() as conn:
        conn.execute(stmt)
    yield engine
    with engine.begin() as conn:
        conn.execute(stmt)


@pytest.fixture()
def db(clean_db: sa.Engine) -> Iterator[Session]:
    with Session(clean_db) as session:
        yield session
