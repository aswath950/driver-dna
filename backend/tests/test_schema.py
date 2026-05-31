"""Phase 2 schema assertions. Requires a running Postgres at DATABASE_URL_SYNC
with migration 0001 applied (``alembic upgrade head``).

These tests inspect the *live* schema via ``sa.inspect`` rather than just the
ORM metadata, so they catch drift between models and migrations.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa

from app.core.config import settings

EXPECTED_TABLES = {
    "seasons",
    "circuits",
    "events",
    "sessions",
    "teams",
    "drivers",
    "session_drivers",
    "race_results",
    "lap_times",
    "driver_stats",
}

EXPECTED_ENUMS = {
    "session_type": {"FP1", "FP2", "FP3", "Q", "SQ", "S", "R"},
    "compound_type": {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN"},
}


@pytest.fixture(scope="module")
def engine() -> sa.Engine:
    eng = sa.create_engine(settings.DATABASE_URL_SYNC)
    try:
        with eng.connect():
            pass
    except sa.exc.OperationalError as e:
        pytest.skip(f"Postgres not reachable: {e}")
    return eng


def test_all_tables_exist(engine: sa.Engine) -> None:
    inspector = sa.inspect(engine)
    actual = set(inspector.get_table_names())
    missing = EXPECTED_TABLES - actual
    assert not missing, f"missing tables: {missing}"


def test_enum_types_present(engine: sa.Engine) -> None:
    with engine.connect() as conn:
        rows = conn.execute(
            sa.text(
                """
                SELECT t.typname AS name, e.enumlabel AS label
                FROM pg_type t
                JOIN pg_enum e ON t.oid = e.enumtypid
                WHERE t.typname IN ('session_type', 'compound_type')
                """
            )
        ).all()
    found: dict[str, set[str]] = {}
    for name, label in rows:
        found.setdefault(name, set()).add(label)
    for name, labels in EXPECTED_ENUMS.items():
        assert found.get(name) == labels, f"enum {name}: got {found.get(name)}"


def test_lap_times_unique_constraint(engine: sa.Engine) -> None:
    inspector = sa.inspect(engine)
    uqs = {u["name"] for u in inspector.get_unique_constraints("lap_times")}
    assert "uq_lap_times_session_driver_lap" in uqs


def test_race_results_unique_constraint(engine: sa.Engine) -> None:
    inspector = sa.inspect(engine)
    uqs = {u["name"] for u in inspector.get_unique_constraints("race_results")}
    assert "uq_race_results_session_driver" in uqs


def test_lap_times_cascade_on_session(engine: sa.Engine) -> None:
    inspector = sa.inspect(engine)
    fks = inspector.get_foreign_keys("lap_times")
    session_fk = next(fk for fk in fks if fk["referred_table"] == "sessions")
    assert session_fk["options"].get("ondelete", "").upper() == "CASCADE"


def test_lap_times_restrict_on_driver(engine: sa.Engine) -> None:
    inspector = sa.inspect(engine)
    fks = inspector.get_foreign_keys("lap_times")
    driver_fk = next(fk for fk in fks if fk["referred_table"] == "drivers")
    assert driver_fk["options"].get("ondelete", "").upper() == "RESTRICT"


def test_driver_stats_composite_pk(engine: sa.Engine) -> None:
    inspector = sa.inspect(engine)
    pk = inspector.get_pk_constraint("driver_stats")
    assert set(pk["constrained_columns"]) == {"driver_id", "season_id"}


def test_session_drivers_composite_pk(engine: sa.Engine) -> None:
    inspector = sa.inspect(engine)
    pk = inspector.get_pk_constraint("session_drivers")
    assert set(pk["constrained_columns"]) == {"session_id", "driver_id"}


def test_seasons_year_unique(engine: sa.Engine) -> None:
    inspector = sa.inspect(engine)
    uqs = {u["name"] for u in inspector.get_unique_constraints("seasons")}
    assert "uq_seasons_year" in uqs


def test_lap_number_check_constraint(engine: sa.Engine) -> None:
    inspector = sa.inspect(engine)
    cks = {c["name"] for c in inspector.get_check_constraints("lap_times")}
    assert "ck_lap_times_lap_number_positive" in cks
