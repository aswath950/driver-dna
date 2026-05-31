"""End-to-end ETL test with mocked OpenF1 HTTP.

We patch the OpenF1Client's underlying ``requests`` calls with the
``responses`` lib, then run ``run()`` against the real Postgres DB,
asserting row counts and — critically — that a second run is a no-op.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import responses
import sqlalchemy as sa

from app.db.models import CompoundType, SessionType
from app.etl import hydrate_session, refresh_driver_stats
from app.etl.hydrate_session import _points_for, _to_compound, _to_session_type

OPENF1_BASE = "https://api.openf1.org/v1"

# Two drivers, 5 laps each, both in 1 stint of compound MEDIUM.
_MEETING_KEY = 999
_SESSION_KEY_RACE = 12345

_MOCK_MEETINGS = [
    {
        "meeting_key": _MEETING_KEY,
        "meeting_name": "Synthetic Grand Prix",
        "year": 2099,
    }
]
_MOCK_SESSIONS = [
    {
        "session_key": _SESSION_KEY_RACE,
        "meeting_key": _MEETING_KEY,
        "meeting_name": "Synthetic Grand Prix",
        "session_name": "Race",
        "session_type": "Race",
        "date_start": "2099-04-01T15:00:00+00:00",
        "date_end": "2099-04-01T17:00:00+00:00",
    }
]
_MOCK_DRIVERS = [
    {
        "driver_number": 1, "name_acronym": "VER", "full_name": "Max Verstappen",
        "team_name": "Red Bull", "team_colour": "1E40AF",
    },
    {
        "driver_number": 44, "name_acronym": "HAM", "full_name": "Lewis Hamilton",
        "team_name": "Mercedes", "team_colour": "06B6D4",
    },
]
_MOCK_LAPS = [
    {"driver_number": 1,  "lap_number": n, "lap_duration": 78.5 + n * 0.05,
     "is_pit_out_lap": False, "session_key": _SESSION_KEY_RACE,
     "st_speed": 290, "date_start": f"2099-04-01T15:0{n}:00+00:00"}
    for n in range(1, 6)
] + [
    {"driver_number": 44, "lap_number": n, "lap_duration": 79.1 + n * 0.04,
     "is_pit_out_lap": False, "session_key": _SESSION_KEY_RACE,
     "st_speed": 288, "date_start": f"2099-04-01T15:0{n}:00+00:00"}
    for n in range(1, 6)
]
_MOCK_STINTS = [
    {"driver_number": 1,  "stint_number": 1, "compound": "MEDIUM",
     "tyre_age_at_start": 0, "lap_start": 1, "lap_end": 5,
     "session_key": _SESSION_KEY_RACE},
    {"driver_number": 44, "stint_number": 1, "compound": "soft",  # lowercase, tests normalisation
     "tyre_age_at_start": 2, "lap_start": 1, "lap_end": 5,
     "session_key": _SESSION_KEY_RACE},
]
# Two position rows per driver; the LAST one is the final classification.
_MOCK_POSITION = [
    {"driver_number": 1,  "position": 3, "date": "2099-04-01T15:01:00+00:00",
     "session_key": _SESSION_KEY_RACE},
    {"driver_number": 1,  "position": 1, "date": "2099-04-01T17:00:00+00:00",
     "session_key": _SESSION_KEY_RACE},
    {"driver_number": 44, "position": 1, "date": "2099-04-01T15:01:00+00:00",
     "session_key": _SESSION_KEY_RACE},
    {"driver_number": 44, "position": 2, "date": "2099-04-01T17:00:00+00:00",
     "session_key": _SESSION_KEY_RACE},
]


_ROUTES = {
    "meetings": _MOCK_MEETINGS,
    "sessions": _MOCK_SESSIONS,
    "drivers":  _MOCK_DRIVERS,
    "laps":     _MOCK_LAPS,
    "stints":   _MOCK_STINTS,
    "position": _MOCK_POSITION,
}


@pytest.fixture()
def mocked_openf1() -> Iterator[responses.RequestsMock]:
    """Register each endpoint enough times to cover multiple hydrate runs.

    The ``responses`` lib consumes registrations in FIFO order. Tests that
    call hydrate twice (idempotency check) issue 6 requests per run, so we
    register each endpoint many times — but always with the SAME real payload
    so order can't matter.
    """
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        for _ in range(20):  # covers >3 hydrate calls comfortably
            for path, payload in _ROUTES.items():
                rsps.add(responses.GET, f"{OPENF1_BASE}/{path}", json=payload)
        yield rsps


# ---------------------------------------------------------------------------
# Unit-level helpers
# ---------------------------------------------------------------------------


def test_points_table_for_top_10() -> None:
    assert _points_for(1) == 25
    assert _points_for(10) == 1
    assert _points_for(11) == 0
    assert _points_for(None) == 0


def test_session_type_mapping() -> None:
    assert _to_session_type("Race") == SessionType.R
    assert _to_session_type("R") == SessionType.R
    assert _to_session_type("Practice 1") == SessionType.FP1
    assert _to_session_type("Qualifying") == SessionType.Q
    assert _to_session_type("Sprint Qualifying") == SessionType.SQ
    assert _to_session_type("Sprint") == SessionType.S
    assert _to_session_type("nonsense") is None
    assert _to_session_type(None) is None


def test_compound_normalisation() -> None:
    assert _to_compound("MEDIUM") == CompoundType.MEDIUM
    assert _to_compound("soft") == CompoundType.SOFT
    assert _to_compound(" Hard ") == CompoundType.HARD
    assert _to_compound("wet") == CompoundType.WET
    assert _to_compound(None) == CompoundType.UNKNOWN
    assert _to_compound("foo") == CompoundType.UNKNOWN


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def _row_counts(engine: sa.Engine) -> dict[str, int]:
    tables = [
        "seasons", "circuits", "events", "sessions", "teams", "drivers",
        "session_drivers", "race_results", "lap_times", "driver_stats",
    ]
    out: dict[str, int] = {}
    with engine.connect() as conn:
        for t in tables:
            out[t] = int(conn.scalar(sa.text(f"SELECT count(*) FROM {t}")) or 0)
    return out


def test_hydrate_run_inserts_expected_rows(
    clean_db: sa.Engine, mocked_openf1: responses.RequestsMock
) -> None:
    result = hydrate_session.run(
        year=2099, grand_prix="Synthetic Grand Prix", session_type="R"
    )
    assert result.season_id is not None
    assert result.event_id is not None
    assert len(result.session_ids) == 1
    assert result.counts == {
        "session_drivers": 2,
        "lap_times": 10,        # 2 drivers × 5 laps
        "race_results": 2,
    }
    counts = _row_counts(clean_db)
    assert counts["seasons"] == 1
    assert counts["events"] == 1
    assert counts["sessions"] == 1
    assert counts["teams"] == 2
    assert counts["drivers"] == 2
    assert counts["session_drivers"] == 2
    assert counts["lap_times"] == 10
    assert counts["race_results"] == 2


def test_hydrate_is_idempotent(
    clean_db: sa.Engine, mocked_openf1: responses.RequestsMock
) -> None:
    """Running the same hydrate twice must produce identical row counts."""
    hydrate_session.run(year=2099, grand_prix="Synthetic Grand Prix", session_type="R")
    first = _row_counts(clean_db)
    hydrate_session.run(year=2099, grand_prix="Synthetic Grand Prix", session_type="R")
    second = _row_counts(clean_db)
    assert first == second


def test_hydrate_dry_run_writes_nothing(
    clean_db: sa.Engine, mocked_openf1: responses.RequestsMock
) -> None:
    result = hydrate_session.run(
        year=2099, grand_prix="Synthetic Grand Prix", session_type="R", dry_run=True
    )
    assert result.dry_run is True
    counts = _row_counts(clean_db)
    assert all(v == 0 for v in counts.values()), counts


def test_results_pick_final_position(
    clean_db: sa.Engine, mocked_openf1: responses.RequestsMock
) -> None:
    """VER ends 1st (last position row), HAM ends 2nd. Points: 25, 18."""
    hydrate_session.run(year=2099, grand_prix="Synthetic Grand Prix", session_type="R")
    with clean_db.connect() as conn:
        rows = conn.execute(sa.text(
            "SELECT d.code, rr.position, rr.points "
            "FROM race_results rr JOIN drivers d ON d.id = rr.driver_id "
            "ORDER BY rr.position"
        )).all()
    assert [(r[0], r[1], float(r[2])) for r in rows] == [
        ("VER", 1, 25.0),
        ("HAM", 2, 18.0),
    ]


def test_compound_attached_from_stints(
    clean_db: sa.Engine, mocked_openf1: responses.RequestsMock
) -> None:
    """All VER laps should be MEDIUM, all HAM laps SOFT."""
    hydrate_session.run(year=2099, grand_prix="Synthetic Grand Prix", session_type="R")
    with clean_db.connect() as conn:
        rows = conn.execute(sa.text(
            "SELECT d.code, lt.compound::text, count(*) "
            "FROM lap_times lt JOIN drivers d ON d.id = lt.driver_id "
            "GROUP BY d.code, lt.compound ORDER BY d.code"
        )).all()
    assert [(r[0], r[1], int(r[2])) for r in rows] == [
        ("HAM", "SOFT",   5),
        ("VER", "MEDIUM", 5),
    ]


def test_lap_times_converted_to_ms(
    clean_db: sa.Engine, mocked_openf1: responses.RequestsMock
) -> None:
    """First VER lap: lap_duration=78.55s -> 78550ms."""
    hydrate_session.run(year=2099, grand_prix="Synthetic Grand Prix", session_type="R")
    with clean_db.connect() as conn:
        ms = conn.scalar(sa.text(
            "SELECT lt.lap_time_ms FROM lap_times lt "
            "JOIN drivers d ON d.id = lt.driver_id "
            "WHERE d.code = 'VER' AND lt.lap_number = 1"
        ))
    assert ms == 78550


def test_refresh_driver_stats_after_hydrate(
    clean_db: sa.Engine, mocked_openf1: responses.RequestsMock
) -> None:
    hydrate_session.run(year=2099, grand_prix="Synthetic Grand Prix", session_type="R")
    res = refresh_driver_stats.run(season_year=2099)
    assert res.rows_affected == 2
    with clean_db.connect() as conn:
        rows = conn.execute(sa.text(
            "SELECT d.code, ds.wins, ds.podiums, float8(ds.points) "
            "FROM driver_stats ds JOIN drivers d ON d.id = ds.driver_id "
            "ORDER BY d.code"
        )).all()
    by_code: dict[str, tuple[Any, ...]] = {r[0]: tuple(r[1:]) for r in rows}
    assert by_code["VER"] == (1, 1, 25.0)
    assert by_code["HAM"] == (0, 1, 18.0)


def test_hydrate_unknown_session_type_raises(
    clean_db: sa.Engine, mocked_openf1: responses.RequestsMock
) -> None:
    with pytest.raises(ValueError, match="unknown session_type"):
        hydrate_session.run(year=2099, grand_prix="Synthetic Grand Prix", session_type="ZZZ")
