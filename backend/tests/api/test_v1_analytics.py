"""Tests for Phase 7 compute endpoints.

The 4 race-analytics endpoints run entirely off the seeded DB (no live
HTTP). The compare endpoint normally hits OpenF1; we mock that with the
``responses`` lib and verify the contract + the trace pipeline.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
import responses
import sqlalchemy as sa
from fastapi.testclient import TestClient

from app.core.config import settings
from app.services.telemetry_compute import N_POINTS

OPENF1 = "https://api.openf1.org/v1"
# Use an int well outside the seed's openf1_session_keys to avoid collisions.
_FAKE_OPENF1_KEY = 99001
_FAKE_SESSION_ID = 1
_DRV_A_ID = 1
_DRV_B_ID = 5
_CAR_A = 901
_CAR_B = 905
# Row returned by the tri-state session-key check (session_status →
# GET /v1/sessions?session_key=...). Only consulted on a cache miss whose
# /laps came back empty, to tell a stale key from a transient failure; the
# happy-path tests below never reach it, but it's registered as headroom.
_SESSIONS_PAYLOAD = [{"session_key": _FAKE_OPENF1_KEY}]


# ---------------------------------------------------------------------------
# Analytics endpoints (DB-backed, no mocks needed)
# ---------------------------------------------------------------------------


def _session_with_data() -> int:
    """Pick any session_id that has lap_times rows."""
    eng = sa.create_engine(settings.DATABASE_URL_SYNC, future=True)
    with eng.connect() as conn:
        sid = conn.scalar(sa.text(
            "SELECT session_id FROM lap_times GROUP BY session_id ORDER BY count(*) DESC LIMIT 1"
        ))
    assert sid is not None, "seed has no lap_times"
    return int(sid)


def test_rolling_pace_happy(client: TestClient) -> None:
    sid = _session_with_data()
    r = client.get(f"/api/v1/sessions/{sid}/analytics/rolling-pace?window=5")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) > 0
    for row in rows[:5]:
        assert set(row.keys()) == {"driver_id", "lap", "rolling_sec"}
        assert row["rolling_sec"] > 30  # any plausible F1 lap > 30s


def test_rolling_pace_validates_window(client: TestClient) -> None:
    sid = _session_with_data()
    assert client.get(f"/api/v1/sessions/{sid}/analytics/rolling-pace?window=0").status_code == 422
    assert client.get(f"/api/v1/sessions/{sid}/analytics/rolling-pace?window=999").status_code == 422


def test_gap_to_leader_happy(client: TestClient) -> None:
    sid = _session_with_data()
    r = client.get(f"/api/v1/sessions/{sid}/analytics/gap-to-leader")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) > 0
    # Each lap should have at least one driver at gap=0 (the leader).
    by_lap: dict[int, list[float]] = {}
    for row in rows:
        by_lap.setdefault(row["lap"], []).append(row["gap_sec"])
    for lap_n, gaps in by_lap.items():
        assert min(gaps) == 0.0, f"lap {lap_n} has no leader (min gap = {min(gaps)})"


def test_undercuts_returns_list(client: TestClient) -> None:
    sid = _session_with_data()
    r = client.get(f"/api/v1/sessions/{sid}/analytics/undercuts")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    # Seed data has no real strategy events; just assert shape on any returned rows.
    for ev in body:
        assert set(ev.keys()) == {"lap", "attacker_id", "victim_id", "type"}
        assert ev["type"] in {"undercut", "overcut"}


def test_tyre_degradation_happy(client: TestClient) -> None:
    sid = _session_with_data()
    r = client.get(f"/api/v1/sessions/{sid}/analytics/tyre-degradation")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) > 0
    for row in rows[:5]:
        assert row["compound"] in {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN"}
        assert row["laps_in_stint"] >= 1


def test_404_on_unknown_session(client: TestClient) -> None:
    for path in (
        "/analytics/rolling-pace",
        "/analytics/gap-to-leader",
        "/analytics/undercuts",
        "/analytics/tyre-degradation",
    ):
        r = client.get(f"/api/v1/sessions/999999{path}")
        assert r.status_code == 404, path
        assert r.json()["type"].endswith("/not_found")


def test_503_when_session_has_no_lap_data(client: TestClient) -> None:
    """Insert a bare session row with no laps; expect 503 envelope."""
    eng = sa.create_engine(settings.DATABASE_URL_SYNC, future=True)
    with eng.begin() as conn:
        circuit_id = conn.scalar(sa.text("SELECT id FROM circuits LIMIT 1"))
        if circuit_id is None:
            circuit_id = conn.scalar(sa.text(
                "INSERT INTO circuits (name) VALUES ('Empty Test Circuit') RETURNING id"
            ))
        season_id = conn.scalar(sa.text(
            "INSERT INTO seasons (year) VALUES (1899) "
            "ON CONFLICT (year) DO UPDATE SET year=EXCLUDED.year RETURNING id"
        ))
        event_id = conn.scalar(sa.text(
            "INSERT INTO events (season_id, circuit_id, round, name, start_date) "
            "VALUES (:s, :c, 99, 'Empty Event', '1899-01-01') "
            "ON CONFLICT (season_id, round) DO UPDATE SET name=EXCLUDED.name RETURNING id"
        ), {"s": season_id, "c": circuit_id})
        empty_session_id = conn.scalar(sa.text(
            "INSERT INTO sessions (event_id, type, date_start, openf1_session_key) "
            "VALUES (:e, 'R', '1899-01-01 12:00:00+00', NULL) RETURNING id"
        ), {"e": event_id})
    try:
        r = client.get(
            f"/api/v1/sessions/{empty_session_id}/analytics/rolling-pace"
        )
        assert r.status_code == 503
        body = r.json()
        assert body["type"].endswith("/upstream_error")
        assert "hydrate" in (body.get("detail") or "")
    finally:
        with eng.begin() as conn:
            conn.execute(sa.text("DELETE FROM sessions WHERE id = :i"),
                         {"i": empty_session_id})


# ---------------------------------------------------------------------------
# Compare endpoint (mocks OpenF1)
# ---------------------------------------------------------------------------


def _make_laps_payload(driver_number: int) -> list[dict]:
    """One full session of synthetic laps with one obvious fastest lap (lap 3).

    Includes ``duration_sector_1/2/3`` so sector-times tests can validate them.
    Fastest lap (lap 3) has duration 78.0 s = 25 + 28 + 25.
    """
    base = 80.0
    return [
        {
            "driver_number": driver_number,
            "lap_number": n,
            "lap_duration": base + (0.5 if n != 3 else -2.0),
            "is_pit_out_lap": False,
            "session_key": _FAKE_OPENF1_KEY,
            "date_start": f"2099-04-01T15:0{n}:00+00:00",
            "duration_sector_1": 25.0,
            "duration_sector_2": 28.0,
            "duration_sector_3": 25.0,
        }
        for n in range(1, 6)
    ]


def _make_car_data(driver_number: int) -> list[dict]:
    """One sample every 100ms across the fastest lap window (lap 3).
    Lap 3 runs from 15:03:00 to ~15:03:78."""
    samples = []
    for i in range(800):  # 800 × 100ms = 80s window
        ms = i * 100
        ts = f"2099-04-01T15:03:{ms // 1000:02d}.{ms % 1000:03d}+00:00"
        samples.append({
            "driver_number": driver_number,
            "date": ts,
            "speed": 250.0 + (driver_number % 3) * 10,  # varies per driver
            "throttle": 100.0,
            "brake": 0.0,
            "rpm": 12000.0,
            "n_gear": 7.0,
            "drs": 0.0,
            "session_key": _FAKE_OPENF1_KEY,
        })
    return samples


def _make_location_data(driver_number: int) -> list[dict]:
    """100 x/y/z location samples within the fastest lap window (lap 3: 15:03:xx)."""
    samples = []
    for i in range(100):
        ms = i * 800  # 800 ms intervals → spans ~80 s
        ts = f"2099-04-01T15:03:{ms // 1000:02d}.{ms % 1000:03d}+00:00"
        samples.append({
            "driver_number": driver_number,
            "date": ts,
            "x": float(i * 10 + driver_number),
            "y": float(i * 5 + driver_number),
            "z": 0.0,
            "session_key": _FAKE_OPENF1_KEY,
        })
    return samples


@pytest.fixture()
def compare_session(client: TestClient) -> Iterator[tuple[int, int, int]]:
    """Spin up a temporary session with two drivers; tear down at end.

    Seeds a dedicated circuit with deterministic ``sector_fractions``
    and outline ``x``/``y`` so the cumtime-based sector logic and the
    track-map figure have known inputs regardless of the rest of the DB.
    """
    eng = sa.create_engine(settings.DATABASE_URL_SYNC, future=True)
    with eng.begin() as conn:
        # Dedicated circuit so sector_fractions and x/y are deterministic.
        circuit_id = conn.scalar(sa.text(
            "INSERT INTO circuits (name, sector_fractions, x, y) "
            "VALUES ('Compare Test Circuit', "
            "        CAST('[0.35, 0.72]' AS JSONB), "
            "        CAST(:x AS JSONB), "
            "        CAST(:y AS JSONB)) "
            "ON CONFLICT (name) DO UPDATE "
            "  SET sector_fractions = EXCLUDED.sector_fractions, "
            "      x = EXCLUDED.x, "
            "      y = EXCLUDED.y "
            "RETURNING id"
        ), {
            "x": json.dumps([float(i) for i in range(10)]),
            "y": json.dumps([float(i * 2) for i in range(10)]),
        })
        season_id = conn.scalar(sa.text(
            "INSERT INTO seasons (year) VALUES (2098) "
            "ON CONFLICT (year) DO UPDATE SET year=EXCLUDED.year RETURNING id"
        ))
        event_id = conn.scalar(sa.text(
            "INSERT INTO events (season_id, circuit_id, round, name, start_date) "
            "VALUES (:s, :c, 99, 'Compare Test', '2098-01-01') "
            "ON CONFLICT (season_id, round) DO UPDATE SET name=EXCLUDED.name RETURNING id"
        ), {"s": season_id, "c": circuit_id})
        session_id = conn.scalar(sa.text(
            "INSERT INTO sessions (event_id, type, date_start, openf1_session_key) "
            "VALUES (:e, 'R', '2098-01-01 12:00:00+00', :k) RETURNING id"
        ), {"e": event_id, "k": _FAKE_OPENF1_KEY})
        team_id = conn.scalar(sa.text(
            "INSERT INTO teams (name) VALUES ('Compare Test Team') "
            "ON CONFLICT (name) DO UPDATE SET name=EXCLUDED.name RETURNING id"
        ))
        drv_a = conn.scalar(sa.text(
            "INSERT INTO drivers (code, full_name) VALUES ('TAA', 'Test A') "
            "ON CONFLICT (code) DO UPDATE SET full_name=EXCLUDED.full_name RETURNING id"
        ))
        drv_b = conn.scalar(sa.text(
            "INSERT INTO drivers (code, full_name) VALUES ('TBB', 'Test B') "
            "ON CONFLICT (code) DO UPDATE SET full_name=EXCLUDED.full_name RETURNING id"
        ))
        for did, car in ((drv_a, _CAR_A), (drv_b, _CAR_B)):
            conn.execute(sa.text(
                "INSERT INTO session_drivers (session_id, driver_id, team_id, car_number) "
                "VALUES (:s, :d, :t, :n) "
                "ON CONFLICT (session_id, driver_id) DO UPDATE SET car_number = EXCLUDED.car_number"
            ), {"s": session_id, "d": did, "t": team_id, "n": car})
    try:
        yield session_id, drv_a, drv_b
    finally:
        with eng.begin() as conn:
            conn.execute(sa.text("DELETE FROM session_drivers WHERE session_id=:s"), {"s": session_id})
            conn.execute(sa.text("DELETE FROM sessions WHERE id=:s"), {"s": session_id})


def test_compare_happy(client: TestClient, compare_session: tuple[int, int, int]) -> None:
    sid, drv_a, drv_b = compare_session
    laps_payload = _make_laps_payload(_CAR_A) + _make_laps_payload(_CAR_B)
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        # responses pops registrations FIFO; we register each endpoint twice
        # (laps once, car_data once per driver) plus some headroom.
        for _ in range(3):
            rsps.add(responses.GET, f"{OPENF1}/sessions", json=_SESSIONS_PAYLOAD)
            rsps.add(responses.GET, f"{OPENF1}/laps", json=laps_payload)
            rsps.add(responses.GET, f"{OPENF1}/car_data", json=_make_car_data(_CAR_A))
            rsps.add(responses.GET, f"{OPENF1}/car_data", json=_make_car_data(_CAR_B))

        r = client.get(
            f"/api/v1/sessions/{sid}/compare?driver_a={drv_a}&driver_b={drv_b}&channel=Speed"
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["channel"] == "Speed"
    assert body["session_id"] == sid
    assert body["driver_a"]["car_number"] == _CAR_A
    assert body["driver_b"]["car_number"] == _CAR_B
    assert body["driver_a"]["code"] == "TAA"
    assert body["driver_b"]["code"] == "TBB"
    assert len(body["driver_a"]["trace"]) == N_POINTS
    assert len(body["driver_b"]["trace"]) == N_POINTS
    # Trace values from synthetic data: constant speed 250 or 251.
    assert all(abs(v - body["driver_a"]["trace"][0]) < 1e-6 for v in body["driver_a"]["trace"])
    # figure_json must be valid Plotly JSON
    fig = json.loads(body["figure_json"])
    assert "data" in fig and "layout" in fig
    assert len(fig["data"]) == 2  # one trace per driver


def test_compare_caches_null_bearing_car_data(
    client: TestClient, compare_session: tuple[int, int, int]
) -> None:
    """Real OpenF1 car_data carries nulls (missing samples / columns). The
    write-through cache must serialize those to JSONB safely and the request
    must still succeed — regression test for the NaN/pd.NA JSONB write that
    used to poison the request transaction and 500."""
    def _null_bearing(driver_number: int) -> list[dict]:
        rows = _make_car_data(driver_number)
        rows[1]["speed"] = None          # a dropped/void sample → NaN
        for r in rows:
            r.pop("drs", None)           # a fully-missing column → pd.NA on _clean
        return rows

    laps_payload = _make_laps_payload(_CAR_A) + _make_laps_payload(_CAR_B)
    sid, drv_a, drv_b = compare_session
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        for _ in range(3):
            rsps.add(responses.GET, f"{OPENF1}/sessions", json=_SESSIONS_PAYLOAD)
            rsps.add(responses.GET, f"{OPENF1}/laps", json=laps_payload)
            rsps.add(responses.GET, f"{OPENF1}/car_data", json=_null_bearing(_CAR_A))
            rsps.add(responses.GET, f"{OPENF1}/car_data", json=_null_bearing(_CAR_B))
        r = client.get(
            f"/api/v1/sessions/{sid}/compare?driver_a={drv_a}&driver_b={drv_b}&channel=Speed"
        )
    # The whole point: no 500 from a poisoned transaction.
    assert r.status_code == 200, r.text
    assert len(r.json()["driver_a"]["trace"]) == N_POINTS


def test_compare_rejects_same_driver(
    client: TestClient, compare_session: tuple[int, int, int]
) -> None:
    sid, drv_a, _ = compare_session
    r = client.get(
        f"/api/v1/sessions/{sid}/compare?driver_a={drv_a}&driver_b={drv_a}&channel=Speed"
    )
    assert r.status_code == 400
    assert "differ" in r.json()["detail"]


def test_compare_unknown_session(client: TestClient) -> None:
    r = client.get(
        "/api/v1/sessions/999999/compare?driver_a=1&driver_b=2&channel=Speed"
    )
    assert r.status_code == 404


def test_compare_invalid_channel(
    client: TestClient, compare_session: tuple[int, int, int]
) -> None:
    sid, drv_a, drv_b = compare_session
    r = client.get(
        f"/api/v1/sessions/{sid}/compare?driver_a={drv_a}&driver_b={drv_b}&channel=Banana"
    )
    assert r.status_code == 422  # FastAPI Query validation


def test_compare_driver_not_in_session(
    client: TestClient, compare_session: tuple[int, int, int]
) -> None:
    sid, drv_a, _ = compare_session
    # Session-key resolution no longer makes a live OpenF1 call, and driver
    # resolution fails before any cache-miss fetch, so this 404 path touches no
    # OpenF1 endpoint at all. The mock is registered only as defensive headroom.
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        rsps.add(responses.GET, f"{OPENF1}/sessions", json=_SESSIONS_PAYLOAD)
        r = client.get(
            f"/api/v1/sessions/{sid}/compare?driver_a={drv_a}&driver_b=999999&channel=Speed"
        )
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Sector Times endpoint
# ---------------------------------------------------------------------------


@pytest.fixture()
def sector_session(client: TestClient) -> Iterator[tuple[int, int, int]]:
    """Session + two drivers + lap_times rows with sector data."""
    eng = sa.create_engine(settings.DATABASE_URL_SYNC, future=True)
    with eng.begin() as conn:
        circuit_id = conn.scalar(sa.text("SELECT id FROM circuits LIMIT 1"))
        season_id = conn.scalar(sa.text(
            "INSERT INTO seasons (year) VALUES (2097) "
            "ON CONFLICT (year) DO UPDATE SET year=EXCLUDED.year RETURNING id"
        ))
        event_id = conn.scalar(sa.text(
            "INSERT INTO events (season_id, circuit_id, round, name, start_date) "
            "VALUES (:s, :c, 98, 'Sector Test', '2097-01-01') "
            "ON CONFLICT (season_id, round) DO UPDATE SET name=EXCLUDED.name RETURNING id"
        ), {"s": season_id, "c": circuit_id})
        session_id = conn.scalar(sa.text(
            "INSERT INTO sessions (event_id, type, date_start) "
            "VALUES (:e, 'R', '2097-01-01 12:00:00+00') RETURNING id"
        ), {"e": event_id})
        team_id = conn.scalar(sa.text(
            "INSERT INTO teams (name) VALUES ('Sector Test Team') "
            "ON CONFLICT (name) DO UPDATE SET name=EXCLUDED.name RETURNING id"
        ))
        drv_a = conn.scalar(sa.text(
            "INSERT INTO drivers (code, full_name) VALUES ('STA', 'Sector A') "
            "ON CONFLICT (code) DO UPDATE SET full_name=EXCLUDED.full_name RETURNING id"
        ))
        drv_b = conn.scalar(sa.text(
            "INSERT INTO drivers (code, full_name) VALUES ('STB', 'Sector B') "
            "ON CONFLICT (code) DO UPDATE SET full_name=EXCLUDED.full_name RETURNING id"
        ))
        for did, car in ((drv_a, 801), (drv_b, 802)):
            conn.execute(sa.text(
                "INSERT INTO session_drivers (session_id, driver_id, team_id, car_number) "
                "VALUES (:s, :d, :t, :n) "
                "ON CONFLICT (session_id, driver_id) DO UPDATE SET car_number = EXCLUDED.car_number"
            ), {"s": session_id, "d": did, "t": team_id, "n": car})
        # Insert lap_times with sector data for each driver.
        for did in (drv_a, drv_b):
            conn.execute(sa.text(
                "INSERT INTO lap_times "
                "(session_id, driver_id, lap_number, lap_time_ms, sector1_ms, sector2_ms, sector3_ms, is_pit_out, is_pit_in) "
                "VALUES (:s, :d, 1, 90000, 28000, 31000, 31000, false, false)"
            ), {"s": session_id, "d": did})
    try:
        yield session_id, drv_a, drv_b
    finally:
        with eng.begin() as conn:
            conn.execute(sa.text("DELETE FROM lap_times WHERE session_id=:s"), {"s": session_id})
            conn.execute(sa.text("DELETE FROM session_drivers WHERE session_id=:s"), {"s": session_id})
            conn.execute(sa.text("DELETE FROM sessions WHERE id=:s"), {"s": session_id})


def test_sector_times_happy(
    client: TestClient, sector_session: tuple[int, int, int]
) -> None:
    sid, drv_a, drv_b = sector_session
    r = client.get(
        f"/api/v1/sessions/{sid}/sector-times?driver_a={drv_a}&driver_b={drv_b}"
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["session_id"] == sid
    assert body["driver_a"]["code"] == "STA"
    assert body["driver_b"]["code"] == "STB"
    assert body["driver_a"]["sector1_ms"] == 28000
    assert body["driver_b"]["sector3_ms"] == 31000
    fig = json.loads(body["figure_json"])
    assert "data" in fig and "layout" in fig
    assert len(fig["data"]) == 2


def test_sector_times_unknown_session(client: TestClient) -> None:
    r = client.get("/api/v1/sessions/999999/sector-times?driver_a=1&driver_b=2")
    assert r.status_code == 404


def test_sector_times_unknown_driver(
    client: TestClient, sector_session: tuple[int, int, int]
) -> None:
    sid, drv_a, _ = sector_session
    r = client.get(
        f"/api/v1/sessions/{sid}/sector-times?driver_a={drv_a}&driver_b=999999"
    )
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Extended /compare channels: RPM, nGear, DRS, TimeDelta
# ---------------------------------------------------------------------------


def _mocked_compare(
    client: TestClient,
    sid: int,
    drv_a: int,
    drv_b: int,
    channel: str,
) -> "responses.RequestsMock":
    """Helper: fire a /compare request with full OpenF1 mocks."""
    laps_payload = _make_laps_payload(_CAR_A) + _make_laps_payload(_CAR_B)
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        for _ in range(3):
            rsps.add(responses.GET, f"{OPENF1}/sessions", json=_SESSIONS_PAYLOAD)
            rsps.add(responses.GET, f"{OPENF1}/laps", json=laps_payload)
            rsps.add(responses.GET, f"{OPENF1}/car_data", json=_make_car_data(_CAR_A))
            rsps.add(responses.GET, f"{OPENF1}/car_data", json=_make_car_data(_CAR_B))
        r = client.get(
            f"/api/v1/sessions/{sid}/compare"
            f"?driver_a={drv_a}&driver_b={drv_b}&channel={channel}"
        )
    return r


def test_compare_rpm(
    client: TestClient, compare_session: tuple[int, int, int]
) -> None:
    sid, drv_a, drv_b = compare_session
    r = _mocked_compare(client, sid, drv_a, drv_b, "RPM")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["channel"] == "RPM"
    assert len(body["driver_a"]["trace"]) == N_POINTS
    fig = json.loads(body["figure_json"])
    assert len(fig["data"]) == 2


def test_compare_ngear(
    client: TestClient, compare_session: tuple[int, int, int]
) -> None:
    sid, drv_a, drv_b = compare_session
    r = _mocked_compare(client, sid, drv_a, drv_b, "nGear")
    assert r.status_code == 200, r.text
    assert r.json()["channel"] == "nGear"


def test_compare_drs(
    client: TestClient, compare_session: tuple[int, int, int]
) -> None:
    sid, drv_a, drv_b = compare_session
    r = _mocked_compare(client, sid, drv_a, drv_b, "DRS")
    assert r.status_code == 200, r.text
    assert r.json()["channel"] == "DRS"


def test_compare_time_delta(
    client: TestClient, compare_session: tuple[int, int, int]
) -> None:
    sid, drv_a, drv_b = compare_session
    laps_payload = _make_laps_payload(_CAR_A) + _make_laps_payload(_CAR_B)
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        # TimeDelta calls /laps once and /car_data twice per driver (Speed × 2)
        for _ in range(4):
            rsps.add(responses.GET, f"{OPENF1}/sessions", json=_SESSIONS_PAYLOAD)
            rsps.add(responses.GET, f"{OPENF1}/laps", json=laps_payload)
            rsps.add(responses.GET, f"{OPENF1}/car_data", json=_make_car_data(_CAR_A))
            rsps.add(responses.GET, f"{OPENF1}/car_data", json=_make_car_data(_CAR_B))
        r = client.get(
            f"/api/v1/sessions/{sid}/compare"
            f"?driver_a={drv_a}&driver_b={drv_b}&channel=TimeDelta"
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["channel"] == "TimeDelta"
    # TimeDelta payload's trace fields now carry per-driver cumtime arrays.
    assert len(body["driver_a"]["trace"]) == N_POINTS
    fig = json.loads(body["figure_json"])
    # Streamlit-style figure has many traces: 2 shaded fills + 2 coloured
    # leader lines + optional lead-change / peak markers.
    assert len(fig["data"]) >= 4
    assert fig["layout"]["title"]["text"].startswith("Lap Time Delta")
    # Three sector vertical lines (S1 at 0%, S2 at sector_fractions[0]*100, S3 at [1]*100).
    shapes = fig["layout"].get("shapes", [])
    vlines = [s for s in shapes if s.get("type") == "line"]
    assert len(vlines) >= 3


def test_compare_sector_times_rejected_from_compare_endpoint(
    client: TestClient, compare_session: tuple[int, int, int]
) -> None:
    sid, drv_a, drv_b = compare_session
    r = client.get(
        f"/api/v1/sessions/{sid}/compare"
        f"?driver_a={drv_a}&driver_b={drv_b}&channel=SectorTimes"
    )
    assert r.status_code == 422  # not in _TelemetryChannel


# ---------------------------------------------------------------------------
# /compare/sectors and /compare/track-map
# ---------------------------------------------------------------------------


def test_compare_sectors_happy(
    client: TestClient, compare_session: tuple[int, int, int]
) -> None:
    sid, drv_a, drv_b = compare_session
    laps_payload = _make_laps_payload(_CAR_A) + _make_laps_payload(_CAR_B)
    # Sectors now come from cumtime × sector_fractions (not from OpenF1
    # duration_sector_*), so the service fetches /car_data too.
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        for _ in range(3):
            rsps.add(responses.GET, f"{OPENF1}/sessions", json=_SESSIONS_PAYLOAD)
            rsps.add(responses.GET, f"{OPENF1}/laps", json=laps_payload)
            rsps.add(responses.GET, f"{OPENF1}/car_data", json=_make_car_data(_CAR_A))
            rsps.add(responses.GET, f"{OPENF1}/car_data", json=_make_car_data(_CAR_B))
        r = client.get(
            f"/api/v1/sessions/{sid}/compare/sectors?driver_a={drv_a}&driver_b={drv_b}"
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["session_id"] == sid
    assert body["driver_a"]["code"] == "TAA"
    assert body["driver_b"]["code"] == "TBB"
    # Fastest lap is 78.0 s. Synthetic car_data has constant speed → cumtime
    # is linear over distance. Compare Test Circuit has sector_fractions
    # [0.35, 0.72]; splits are taken at grid indices i1 = round(0.35*(N-1)) and
    # i2 = round(0.72*(N-1)) over the N_POINTS grid, then scaled by lap time.
    n_1 = N_POINTS - 1
    i1 = round(0.35 * n_1)
    i2 = round(0.72 * n_1)
    exp_s1 = round(78_000 * i1 / n_1)
    exp_s2 = round(78_000 * (i2 - i1) / n_1)
    exp_s3 = round(78_000 * (n_1 - i2) / n_1)
    splits_a = body["driver_a"]
    assert abs(splits_a["sector1_ms"] - exp_s1) <= 1
    assert abs(splits_a["sector2_ms"] - exp_s2) <= 1
    assert abs(splits_a["sector3_ms"] - exp_s3) <= 1
    # Sectors sum to the official lap duration (78.000s → 78_000 ms).
    total = splits_a["sector1_ms"] + splits_a["sector2_ms"] + splits_a["sector3_ms"]
    assert abs(total - 78_000) <= 1
    fig = json.loads(body["figure_json"])
    assert len(fig["data"]) == 2
    assert fig["layout"]["yaxis"]["title"]["text"] == "Time (s)"


def test_compare_sectors_unknown_driver(
    client: TestClient, compare_session: tuple[int, int, int]
) -> None:
    sid, drv_a, _ = compare_session
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        rsps.add(responses.GET, f"{OPENF1}/sessions", json=_SESSIONS_PAYLOAD)
        r = client.get(
            f"/api/v1/sessions/{sid}/compare/sectors?driver_a={drv_a}&driver_b=999999"
        )
    assert r.status_code == 404


def test_compare_track_map_happy(
    client: TestClient, compare_session: tuple[int, int, int]
) -> None:
    sid, drv_a, drv_b = compare_session
    laps_payload = _make_laps_payload(_CAR_A) + _make_laps_payload(_CAR_B)
    # Track map now uses circuit outline from Postgres + per-driver Speed
    # traces from /car_data; /location is no longer called.
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        for _ in range(3):
            rsps.add(responses.GET, f"{OPENF1}/sessions", json=_SESSIONS_PAYLOAD)
            rsps.add(responses.GET, f"{OPENF1}/laps", json=laps_payload)
            rsps.add(responses.GET, f"{OPENF1}/car_data", json=_make_car_data(_CAR_A))
            rsps.add(responses.GET, f"{OPENF1}/car_data", json=_make_car_data(_CAR_B))
        r = client.get(
            f"/api/v1/sessions/{sid}/compare/track-map"
            f"?driver_a={drv_a}&driver_b={drv_b}"
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["session_id"] == sid
    assert body["driver_a"]["code"] == "TAA"
    assert body["driver_b"]["code"] == "TBB"
    # Circuit outline from the seeded "Compare Test Circuit" (10 points).
    assert len(body["circuit_x"]) == 10
    assert len(body["circuit_y"]) == 10
    fig = json.loads(body["figure_json"])
    # Streamlit-style track map: background outline + ≥1 winner segment + 3 sector markers.
    assert len(fig["data"]) >= 5
    assert fig["layout"]["title"]["text"].startswith("Track Map")


def test_compare_track_map_unknown_session(client: TestClient) -> None:
    r = client.get(
        "/api/v1/sessions/999999/compare/track-map?driver_a=1&driver_b=2"
    )
    assert r.status_code == 404
