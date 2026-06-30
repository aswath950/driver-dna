"""Telemetry fetch ETL tests.

We seed a minimal session row, mock OpenF1 HTTP with the ``responses`` lib, and
assert:
  * ``fetch_telemetry.run`` distinguishes a stale (404) session_key from a
    session that simply has no laps; and
  * a re-fetch wipes any previously-cached telemetry and starts fresh, so stale
    rows from a prior run never linger.
Runs against the real Postgres DB via the shared ``clean_db`` fixture.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
import responses
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.db.models import (
    CarTelemetry,
    Circuit,
    Driver,
    Event,
    Season,
    SessionDriver,
    SessionType,
    Team,
)
from app.db.models import Session as SessionRow
from app.etl import fetch_telemetry

OPENF1_BASE = "https://api.openf1.org/v1"

_STALE_KEY = 960  # below every valid OpenF1 session_key range (~7763-10033)


def _seed_session(engine: sa.Engine, *, session_key: int) -> int:
    """Insert season → circuit → event → session and return the session id."""
    with Session(engine) as db:
        season = Season(year=2099)
        circuit = Circuit(name="Synthetic Circuit")
        db.add_all([season, circuit])
        db.flush()
        event = Event(
            season_id=season.id,
            circuit_id=circuit.id,
            round=1,
            name="Synthetic Grand Prix",
        )
        db.add(event)
        db.flush()
        srow = SessionRow(
            event_id=event.id,
            type=SessionType.R,
            date_start=datetime(2099, 4, 1, 15, 0, tzinfo=timezone.utc),
            openf1_session_key=session_key,
        )
        db.add(srow)
        db.flush()
        sid = srow.id
        db.commit()
    return sid


@patch("src.openf1.time.sleep")  # skip retry backoff sleeps for the 404 path
@responses.activate
def test_stale_session_key_reports_actionable_error(
    mock_sleep, clean_db: sa.Engine
) -> None:
    sid = _seed_session(clean_db, session_key=_STALE_KEY)

    # Laps come back empty; the session no longer exists (404) → stale key.
    responses.add(responses.GET, f"{OPENF1_BASE}/laps", json=[], status=200)
    responses.add(
        responses.GET, f"{OPENF1_BASE}/sessions",
        json={"detail": "No results found."}, status=404,
    )

    result = fetch_telemetry.run(session_id=sid)

    assert result.drivers_processed == 0
    assert result.laps_stored == 0
    assert len(result.errors) == 1
    assert "stale key" in result.errors[0]
    assert str(_STALE_KEY) in result.errors[0]


@responses.activate
def test_valid_key_with_no_laps_keeps_original_message(
    clean_db: sa.Engine,
) -> None:
    sid = _seed_session(clean_db, session_key=_STALE_KEY)

    # Laps empty BUT the session exists → genuine no-laps case, not a stale key.
    responses.add(responses.GET, f"{OPENF1_BASE}/laps", json=[], status=200)
    responses.add(
        responses.GET, f"{OPENF1_BASE}/sessions",
        json=[{"session_key": _STALE_KEY, "session_name": "Race"}], status=200,
    )

    result = fetch_telemetry.run(session_id=sid)

    assert result.drivers_processed == 0
    assert len(result.errors) == 1
    assert result.errors[0] == f"OpenF1 returned no laps for session_key={_STALE_KEY}"
    assert "stale key" not in result.errors[0]


_VALID_KEY = 9472
_DRIVER_NUMBER = 7


def _seed_session_with_driver(engine: sa.Engine, *, session_key: int) -> tuple[int, int]:
    """Seed season→circuit→event→session plus one driver/session_driver.

    Returns ``(session_id, driver_id)``.
    """
    with Session(engine) as db:
        season = Season(year=2099)
        circuit = Circuit(name="Synthetic Circuit")
        team = Team(name="Synthetic Team")
        db.add_all([season, circuit, team])
        db.flush()
        event = Event(
            season_id=season.id, circuit_id=circuit.id, round=1,
            name="Synthetic Grand Prix",
        )
        driver = Driver(code="SYN", full_name="Syn Driver", current_team_id=team.id)
        db.add_all([event, driver])
        db.flush()
        srow = SessionRow(
            event_id=event.id, type=SessionType.R,
            date_start=datetime(2099, 4, 1, 15, 0, tzinfo=timezone.utc),
            openf1_session_key=session_key,
        )
        db.add(srow)
        db.flush()
        db.add(SessionDriver(
            session_id=srow.id, driver_id=driver.id, team_id=team.id,
            car_number=_DRIVER_NUMBER,
        ))
        sid, did = srow.id, driver.id
        db.commit()
    return sid, did


def _telemetry_rows(engine: sa.Engine, session_id: int) -> list[tuple[int, dict]]:
    with Session(engine) as db:
        rows = db.execute(
            sa.select(CarTelemetry.lap_number, CarTelemetry.samples)
            .where(CarTelemetry.session_id == session_id)
            .order_by(CarTelemetry.lap_number)
        ).all()
    return [(int(r[0]), r[1]) for r in rows]


@responses.activate
def test_refetch_wipes_existing_and_starts_fresh(clean_db: sa.Engine) -> None:
    sid, did = _seed_session_with_driver(clean_db, session_key=_VALID_KEY)

    # Pre-existing cache: lap 5 (will be re-fetched with new data) and a stale
    # lap 99 that the fresh fetch will NOT produce — it must be purged.
    with Session(clean_db) as db:
        db.add_all([
            CarTelemetry(
                session_id=sid, driver_id=did, lap_number=5,
                lap_duration=999.0, samples={"dates": ["OLD"], "speed": [1]},
            ),
            CarTelemetry(
                session_id=sid, driver_id=did, lap_number=99,
                lap_duration=999.0, samples={"dates": ["STALE"], "speed": [2]},
            ),
        ])
        db.commit()

    # Fresh fetch: one fastest lap (5) with three car_data samples in-window.
    responses.add(responses.GET, f"{OPENF1_BASE}/laps", json=[{
        "driver_number": _DRIVER_NUMBER, "lap_number": 5, "lap_duration": 80.0,
        "is_pit_out_lap": False, "session_key": _VALID_KEY,
        "date_start": "2099-04-01T15:01:00+00:00",
    }], status=200)
    responses.add(responses.GET, f"{OPENF1_BASE}/car_data", json=[
        {"driver_number": _DRIVER_NUMBER, "date": f"2099-04-01T15:01:{s:02d}+00:00",
         "speed": 300 + s, "throttle": 100, "brake": 0, "n_gear": 7, "rpm": 11000,
         "drs": 1, "session_key": _VALID_KEY}
        for s in (10, 40, 59)
    ], status=200)

    result = fetch_telemetry.run(session_id=sid)

    assert result.deleted_existing == 2
    assert result.drivers_processed == 1
    assert result.laps_stored == 1

    rows = _telemetry_rows(clean_db, sid)
    # Stale lap 99 is gone; only the freshly-fetched lap 5 remains.
    assert [lap for lap, _ in rows] == [5]
    fresh_samples = rows[0][1]
    assert fresh_samples["dates"] != ["OLD"]           # overwritten, not stale
    assert len(fresh_samples["speed"]) == 3            # the three new samples
