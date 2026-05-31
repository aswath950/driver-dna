"""Phase 3 regression guard. Each query in `docs/query_plans.md` must
continue to use an index-based plan. We parse the JSON form of EXPLAIN
ANALYZE and walk it looking for an `Index*Scan` node anywhere in the tree.

Requires:
- Postgres reachable at `settings.DATABASE_URL_SYNC`
- Migration 0002 applied
- `scripts/seed_demo.sql` loaded (the test seeds it automatically if the
  fact table is empty so this also doubles as a setup check)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from app.core.config import settings

SEED_SQL_PATH = Path(__file__).resolve().parents[1] / "scripts" / "seed_demo.sql"


@pytest.fixture(scope="module")
def engine() -> sa.Engine:
    eng = sa.create_engine(settings.DATABASE_URL_SYNC)
    try:
        with eng.connect():
            pass
    except sa.exc.OperationalError as e:
        pytest.skip(f"Postgres not reachable: {e}")
    return eng


@pytest.fixture(scope="module", autouse=True)
def ensure_seeded(engine: sa.Engine) -> None:
    """Load seed_demo.sql if the lap_times table is empty (or far below the
    full size — e.g. after ETL tests truncated everything)."""
    with engine.connect() as conn:
        n = conn.scalar(sa.text("SELECT COUNT(*) FROM lap_times")) or 0
    if n < 100_000:
        sql = SEED_SQL_PATH.read_text()
        # psycopg supports multi-statement execute via the raw driver cursor,
        # which handles comments and ';' correctly without naive splitting.
        raw = engine.raw_connection()
        try:
            cur = raw.cursor()
            cur.execute(sql)
            raw.commit()
        finally:
            raw.close()


def _walk_plan(node: dict[str, Any]) -> list[str]:
    """Return all Node Type strings in the plan tree."""
    out: list[str] = [node.get("Node Type", "")]
    for child in node.get("Plans", []) or []:
        out.extend(_walk_plan(child))
    return out


def _explain(engine: sa.Engine, sql: str) -> list[str]:
    with engine.connect() as conn:
        row = conn.execute(sa.text(f"EXPLAIN (FORMAT JSON) {sql}")).scalar_one()
    if isinstance(row, str):
        row = json.loads(row)
    plan = row[0]["Plan"]
    return _walk_plan(plan)


def _assert_uses_index(nodes: list[str], query_name: str) -> None:
    assert any("Index" in n and "Scan" in n for n in nodes), (
        f"{query_name}: expected an Index*Scan in plan, got {nodes}"
    )


def test_q1_leaderboard_uses_index(engine: sa.Engine) -> None:
    nodes = _explain(
        engine,
        """
        SELECT rr.position, d.code, rr.points
        FROM race_results rr
        JOIN drivers d ON d.id = rr.driver_id
        WHERE rr.session_id = 36
        ORDER BY rr.position
        """,
    )
    _assert_uses_index(nodes, "Q1 leaderboard")


def test_q2_driver_pace_uses_index(engine: sa.Engine) -> None:
    nodes = _explain(
        engine,
        """
        SELECT lap_number, lap_time_ms
        FROM lap_times
        WHERE session_id = 36 AND driver_id = 5
        ORDER BY lap_number
        """,
    )
    _assert_uses_index(nodes, "Q2 driver pace")


def test_q3_recent_races_uses_index(engine: sa.Engine) -> None:
    nodes = _explain(
        engine,
        """
        SELECT id, name, start_date
        FROM events
        ORDER BY start_date DESC
        LIMIT 10
        """,
    )
    _assert_uses_index(nodes, "Q3 recent races")


def test_required_indexes_exist(engine: sa.Engine) -> None:
    expected = {
        "ix_lap_times_session_lap",
        "ix_lap_times_pit",
        "ix_race_results_session_pos",
        "ix_events_start_date",
        "ix_driver_stats_season_points",
    }
    with engine.connect() as conn:
        rows = conn.execute(
            sa.text(
                "SELECT indexname FROM pg_indexes "
                "WHERE schemaname='public' AND indexname LIKE 'ix_%'"
            )
        ).all()
    actual = {r[0] for r in rows}
    missing = expected - actual
    assert not missing, f"missing indexes: {missing}"
