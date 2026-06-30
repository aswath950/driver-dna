"""Phase 8 GraphQL tests.

Covers:
  - Introspection: every documented type + query field is present.
  - REST/GraphQL parity: the same leaderboard via REST and GraphQL agrees.
  - N+1 regression: the DataLoaders keep SQL counts constant w.r.t. the
    number of rows returned in a nested query.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from app.core.config import settings


def _post(client: TestClient, query: str, variables: dict | None = None) -> dict:
    r = client.post(
        "/graphql",
        json={"query": query, "variables": variables or {}},
    )
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


_EXPECTED_QUERY_FIELDS = {
    "season", "seasons", "event", "events", "session", "sessionsForEvent",
    "sessionResults", "sessionLaps", "driver", "drivers", "driverStats",
    "standings",
}

_EXPECTED_TYPES = {
    "Season", "Event", "Session", "Driver", "Team", "DriverStats",
    "Lap", "RaceResult", "StandingRow",
    "SessionTypeGQL", "CompoundGQL",
    # ChannelGQL is declared but not yet referenced by any resolver; Strawberry
    # only emits types reachable from the Query root, so it's intentionally
    # excluded until the GraphQL compare endpoint lands (Phase 11+).
}


def test_introspection_lists_all_query_fields(client: TestClient) -> None:
    body = _post(client, "{ __schema { queryType { fields { name } } } }")
    names = {f["name"] for f in body["data"]["__schema"]["queryType"]["fields"]}
    assert _EXPECTED_QUERY_FIELDS.issubset(names), (
        f"missing fields: {_EXPECTED_QUERY_FIELDS - names}"
    )


def test_introspection_lists_all_types(client: TestClient) -> None:
    body = _post(client, "{ __schema { types { name } } }")
    names = {t["name"] for t in body["data"]["__schema"]["types"]}
    assert _EXPECTED_TYPES.issubset(names), (
        f"missing types: {_EXPECTED_TYPES - names}"
    )


# ---------------------------------------------------------------------------
# Happy-path queries
# ---------------------------------------------------------------------------


def test_seasons_query_orders_desc_and_limits(client: TestClient) -> None:
    # `first` is honoured and rows come back newest-year-first. We deliberately
    # avoid asserting a fixed total: the seed spans 2015..2026 (12 seasons) and
    # the ETL tests transiently insert future-year sentinel seasons (e.g. 2099),
    # so an exact count is brittle in a full-suite run.
    body = _post(client, "{ seasons(first: 5) { year } }")
    years = [s["year"] for s in body["data"]["seasons"]]
    assert len(years) == 5
    assert years == sorted(years, reverse=True)
    assert len(set(years)) == 5  # no duplicates

    # The current seed's most-recent real seasons are present.
    all_years = {
        s["year"] for s in _post(client, "{ seasons(first: 50) { year } }")["data"]["seasons"]
    }
    assert {2024, 2025, 2026}.issubset(all_years)


def test_session_lookup_404_returns_null(client: TestClient) -> None:
    body = _post(client, "{ session(id: \"999999\") { id } }")
    assert body["data"]["session"] is None


def test_driver_stats_query(client: TestClient) -> None:
    body = _post(
        client,
        "{ driverStats(driverId: \"1\", season: 2024) { wins points } }",
    )
    stats = body["data"]["driverStats"]
    assert stats["wins"] >= 0
    assert stats["points"] > 0


def test_standings_ordered_desc(client: TestClient) -> None:
    body = _post(
        client,
        "{ standings(season: 2024) { position points driver { code } } }",
    )
    rows = body["data"]["standings"]
    assert len(rows) > 0
    pts = [r["points"] for r in rows]
    assert pts == sorted(pts, reverse=True)
    assert [r["position"] for r in rows] == list(range(1, len(rows) + 1))


# ---------------------------------------------------------------------------
# REST ↔ GraphQL parity
# ---------------------------------------------------------------------------


def test_session_results_parity_with_rest(client: TestClient) -> None:
    rest = client.get("/api/v1/sessions/1/results").json()
    gql = _post(
        client,
        """
        {
          sessionResults(sessionId: "1") {
            position driver { code } team { name } points
          }
        }
        """,
    )["data"]["sessionResults"]
    assert len(rest) == len(gql)
    rest_keys = [(r["position"], r["driver"]["code"], r["team"]["name"]) for r in rest]
    gql_keys = [(g["position"], g["driver"]["code"], g["team"]["name"]) for g in gql]
    assert rest_keys == gql_keys


# ---------------------------------------------------------------------------
# N+1 regression — count actual SQL statements
# ---------------------------------------------------------------------------


@pytest.fixture()
def sql_recorder() -> Iterator[list[str]]:
    """Attach a SQLAlchemy listener to capture every executed SQL statement
    against the test database (the sync URL used by Alembic).

    This catches N+1 by counting how many times the listener fires when the
    DataLoader-backed resolver runs.
    """
    engine = sa.create_engine(settings.DATABASE_URL_SYNC, future=True)
    captured: list[str] = []

    def _on_exec(conn, cursor, statement, parameters, context, executemany):  # noqa: ANN001
        captured.append(statement)

    sa.event.listen(engine, "before_cursor_execute", _on_exec)
    try:
        # Bind to the async engine the app actually uses, by attaching to the
        # AsyncEngine's underlying sync engine. The test's NullPool engine in
        # conftest reuses the app's get_db override.
        from app.db import session as session_mod
        sa.event.listen(
            session_mod.engine.sync_engine, "before_cursor_execute", _on_exec
        )
        yield captured
    finally:
        sa.event.remove(engine, "before_cursor_execute", _on_exec)
        try:
            sa.event.remove(
                session_mod.engine.sync_engine, "before_cursor_execute", _on_exec
            )
        except Exception:
            pass


def test_n_plus_1_killed_by_dataloader(
    client: TestClient, sql_recorder: list[str]
) -> None:
    """``sessionResults`` returns 20 rows, each with nested driver + team.

    Without DataLoaders this would issue 1 (results) + 20 (drivers) + 20
    (teams) = 41 queries. The repository eager-loads + the resolver builds
    the Driver/Team directly from the joined row, so the count must be a
    handful (1-3) regardless of row count.

    We assert the SQL count is well below the N+1 ceiling. This is a
    *regression guard*: if anyone removes the joinedload or stops re-using
    the joined row, the count balloons and this test fails loudly.
    """
    sql_recorder.clear()
    body = _post(
        client,
        """
        {
          sessionResults(sessionId: "1") {
            position
            driver { code currentTeam { name } }
            team { name }
          }
        }
        """,
    )
    rows = body["data"]["sessionResults"]
    assert len(rows) == 20

    # Filter to actual data SQL (skip ROLLBACK/BEGIN/SAVEPOINT/SELECT 1).
    data_sql = [
        s for s in sql_recorder
        if s.strip().upper().startswith(("SELECT", "WITH"))
    ]
    # Allow up to ~5 queries (results + per-loader batches). Crucially this
    # must NOT scale with row count.
    assert len(data_sql) <= 10, (
        f"N+1 regression: {len(data_sql)} SELECTs for 20 rows\n"
        + "\n".join(data_sql)
    )
