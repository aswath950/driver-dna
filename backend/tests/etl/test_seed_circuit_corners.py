"""Tests for the ``seed-circuit-corners`` ETL command.

FastF1 is fully mocked: ``get_event_schedule`` returns a synthetic schedule
DataFrame and ``get_session`` returns a fake session whose circuit info
carries a known corner list.  Verifies:

- the FastF1 round is resolved by event-name match against the schedule,
  never by the DB ``events.round`` value (which holds OpenF1 meeting_keys)
- corners + length_km are stored on the circuit row
- unmatched event names are skipped with a recorded error and no DB write
- future-dated events fall back to an earlier season
- running the seeder twice overwrites cleanly (idempotent)
"""

from __future__ import annotations

import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.db.models import Circuit, Event, Season
from app.etl import seed_circuit_corners

_CORNERS = [
    {"number": 1, "letter": "", "distance_m": 140.0},
    {"number": 2, "letter": "", "distance_m": 620.0},
    {"number": 3, "letter": "A", "distance_m": 1480.0},
]
_TOTAL_M = 3337.0


class _FakeCircuitInfo:
    def __init__(self, corners: list[dict], total_m: float) -> None:
        self.corners = pd.DataFrame(
            [
                {"Number": c["number"], "Letter": c["letter"], "Distance": c["distance_m"]}
                for c in corners
            ]
        )
        self.marshal_sectors = pd.DataFrame({"Distance": [total_m / 2, total_m]})


class _FakeSession:
    def __init__(self, corners: list[dict], total_m: float) -> None:
        self._info = _FakeCircuitInfo(corners, total_m)

    def load(self, **kwargs) -> None:
        pass

    def get_circuit_info(self) -> _FakeCircuitInfo:
        return self._info


def _schedule_df(entries: list[tuple[str, int, str]]) -> pd.DataFrame:
    """entries: (event_name, round_number, event_date_iso)."""
    return pd.DataFrame(
        {
            "EventName": [e[0] for e in entries],
            "RoundNumber": [e[1] for e in entries],
            "EventDate": [pd.Timestamp(e[2]) for e in entries],
        }
    )


@pytest.fixture()
def fake_fastf1(monkeypatch: pytest.MonkeyPatch):
    """Patch fastf1 in the seed module; records every get_session call."""
    calls: list[tuple[int, int]] = []
    schedules: dict[int, pd.DataFrame] = {}

    def get_event_schedule(year: int, include_testing: bool = True) -> pd.DataFrame:
        if year not in schedules:
            raise ValueError(f"no schedule for {year}")
        return schedules[year]

    def get_session(year: int, round_num: int, kind: str) -> _FakeSession:
        calls.append((year, round_num))
        return _FakeSession(_CORNERS, _TOTAL_M)

    monkeypatch.setattr(seed_circuit_corners.fastf1, "get_event_schedule", get_event_schedule)
    monkeypatch.setattr(seed_circuit_corners.fastf1, "get_session", get_session)
    monkeypatch.setattr(seed_circuit_corners.fastf1.Cache, "enable_cache", lambda path: None)
    return {"calls": calls, "schedules": schedules}


def _seed_event(
    db: Session,
    *,
    circuit_name: str,
    event_name: str,
    year: int,
    round_: int,
    length_km: float | None = None,
    corners: list[dict] | None = None,
) -> int:
    season = db.query(Season).filter_by(year=year).first()
    if season is None:
        season = Season(year=year)
        db.add(season)
        db.flush()
    circuit = db.query(Circuit).filter_by(name=circuit_name).first()
    if circuit is None:
        circuit = Circuit(name=circuit_name, length_km=length_km, corners=corners)
        db.add(circuit)
        db.flush()
    event = Event(season_id=season.id, circuit_id=circuit.id, round=round_, name=event_name)
    db.add(event)
    db.commit()
    return event.id


def _read_circuit(engine: sa.Engine, name: str) -> dict:
    with engine.connect() as conn:
        row = conn.execute(
            sa.text("SELECT name, length_km, corners FROM circuits WHERE name = :n"),
            {"n": name},
        ).first()
    return dict(row._mapping) if row else {}


def test_resolves_round_by_name_and_stores_corners(
    clean_db: sa.Engine, db: Session, fake_fastf1: dict
) -> None:
    # DB event stores an OpenF1 meeting_key (1286) as round — must be ignored.
    _seed_event(
        db, circuit_name="Monaco Grand Prix", event_name="Monaco Grand Prix",
        year=2025, round_=1286,
    )
    fake_fastf1["schedules"][2025] = _schedule_df(
        [("Bahrain Grand Prix", 1, "2025-04-13"), ("Monaco Grand Prix", 8, "2025-05-25")]
    )

    result = seed_circuit_corners.run()

    assert result.circuits_updated == 1
    assert result.errors == []
    # FastF1 was called with the schedule round, never the meeting_key.
    assert fake_fastf1["calls"] == [(2025, 8)]

    row = _read_circuit(clean_db, "Monaco Grand Prix")
    assert row["corners"] == _CORNERS
    assert float(row["length_km"]) == round(_TOTAL_M / 1000, 3)


def test_unmatched_event_name_is_skipped(
    clean_db: sa.Engine, db: Session, fake_fastf1: dict
) -> None:
    # Generic seed-data names ("Round 1 2024") never match the schedule.
    _seed_event(
        db, circuit_name="Circuit de Monaco", event_name="Round 1 2024",
        year=2024, round_=1,
    )
    fake_fastf1["schedules"][2024] = _schedule_df([("Monaco Grand Prix", 8, "2024-05-26")])
    fake_fastf1["schedules"][2023] = _schedule_df([("Monaco Grand Prix", 7, "2023-05-28")])
    fake_fastf1["schedules"][2022] = _schedule_df([("Monaco Grand Prix", 7, "2022-05-29")])

    result = seed_circuit_corners.run()

    assert result.circuits_updated == 0
    assert result.circuits_skipped == 1
    assert "Circuit de Monaco" in result.errors[0]
    assert fake_fastf1["calls"] == []
    assert _read_circuit(clean_db, "Circuit de Monaco")["corners"] is None


def test_future_event_falls_back_to_earlier_season(
    clean_db: sa.Engine, db: Session, fake_fastf1: dict
) -> None:
    future = (pd.Timestamp.now() + pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    this_year = pd.Timestamp.now().year
    _seed_event(
        db, circuit_name="Belgian Grand Prix", event_name="Belgian Grand Prix",
        year=this_year, round_=1290,
    )
    # This season's event hasn't happened yet; last season's has.
    fake_fastf1["schedules"][this_year] = _schedule_df([("Belgian Grand Prix", 13, future)])
    fake_fastf1["schedules"][this_year - 1] = _schedule_df(
        [("Belgian Grand Prix", 13, f"{this_year - 1}-07-27")]
    )

    result = seed_circuit_corners.run()

    assert result.circuits_updated == 1
    assert fake_fastf1["calls"] == [(this_year - 1, 13)]
    assert _read_circuit(clean_db, "Belgian Grand Prix")["corners"] == _CORNERS


def test_preferred_year_is_tried_first(
    clean_db: sa.Engine, db: Session, fake_fastf1: dict
) -> None:
    _seed_event(
        db, circuit_name="Monaco Grand Prix", event_name="Monaco Grand Prix",
        year=2025, round_=1286,
    )
    fake_fastf1["schedules"][2024] = _schedule_df([("Monaco Grand Prix", 8, "2024-05-26")])
    fake_fastf1["schedules"][2025] = _schedule_df([("Monaco Grand Prix", 8, "2025-05-25")])

    result = seed_circuit_corners.run(year=2024)

    assert result.year == 2024
    assert result.circuits_updated == 1
    assert fake_fastf1["calls"] == [(2024, 8)]


def test_rerun_overwrites_cleanly(
    clean_db: sa.Engine, db: Session, fake_fastf1: dict
) -> None:
    # Stale length_km must be replaced: it normalises the corner distance_m
    # values at runtime, so it has to come from the same FastF1 lap.
    _seed_event(
        db, circuit_name="Monaco Grand Prix", event_name="Monaco Grand Prix",
        year=2025, round_=1286, length_km=9.999,
    )
    fake_fastf1["schedules"][2025] = _schedule_df([("Monaco Grand Prix", 8, "2025-05-25")])

    seed_circuit_corners.run()
    first = _read_circuit(clean_db, "Monaco Grand Prix")
    seed_circuit_corners.run()
    second = _read_circuit(clean_db, "Monaco Grand Prix")

    assert first == second
    assert second["corners"] == _CORNERS
    assert float(second["length_km"]) == round(_TOTAL_M / 1000, 3)


def test_seed_for_event_stores_corners(
    clean_db: sa.Engine, db: Session, fake_fastf1: dict
) -> None:
    # The hydrate hook: a newly hydrated event's circuit gets corner data.
    event_id = _seed_event(
        db, circuit_name="Dutch Grand Prix", event_name="Dutch Grand Prix",
        year=2025, round_=1271,
    )
    fake_fastf1["schedules"][2025] = _schedule_df([("Dutch Grand Prix", 15, "2025-08-31")])

    stored = seed_circuit_corners.seed_for_event(db, event_id)

    assert stored is True
    assert fake_fastf1["calls"] == [(2025, 15)]
    assert _read_circuit(clean_db, "Dutch Grand Prix")["corners"] == _CORNERS


def test_seed_for_event_noop_when_already_seeded(
    clean_db: sa.Engine, db: Session, fake_fastf1: dict
) -> None:
    existing = [{"number": 1, "letter": "", "distance_m": 99.0}]
    event_id = _seed_event(
        db, circuit_name="Dutch Grand Prix", event_name="Dutch Grand Prix",
        year=2025, round_=1271, corners=existing,
    )

    stored = seed_circuit_corners.seed_for_event(db, event_id)

    assert stored is False
    assert fake_fastf1["calls"] == []
    assert _read_circuit(clean_db, "Dutch Grand Prix")["corners"] == existing


def test_seed_for_event_survives_fastf1_failure(
    clean_db: sa.Engine, db: Session, fake_fastf1: dict
) -> None:
    # No schedule registered for any year → every lookup raises inside
    # FastF1; seed_for_event must swallow it and report False.
    event_id = _seed_event(
        db, circuit_name="Dutch Grand Prix", event_name="Dutch Grand Prix",
        year=2025, round_=1271,
    )

    stored = seed_circuit_corners.seed_for_event(db, event_id)

    assert stored is False
    assert _read_circuit(clean_db, "Dutch Grand Prix")["corners"] is None


def test_sentinel_years_are_excluded(
    clean_db: sa.Engine, db: Session, fake_fastf1: dict
) -> None:
    # Sentinel rows like "Empty Event" (1899) and "Sector Test" (2097) must
    # never reach FastF1.
    _seed_event(
        db, circuit_name="Compare Test Circuit", event_name="Compare Test",
        year=2098, round_=99,
    )

    result = seed_circuit_corners.run()

    assert result.circuits_updated == 0
    assert result.circuits_skipped == 0
    assert fake_fastf1["calls"] == []
