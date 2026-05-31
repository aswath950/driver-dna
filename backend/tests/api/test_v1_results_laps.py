"""Tests for /api/v1/sessions/{id}/results and /api/v1/sessions/{id}/laps."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_leaderboard_happy(client: TestClient) -> None:
    r = client.get("/api/v1/sessions/1/results")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert len(body) > 0
    positions = [row["position"] for row in body if row["position"] is not None]
    assert positions == sorted(positions), "ordered by position ASC"
    for row in body[:3]:
        assert row["driver"]["code"]
        assert row["team"]["name"]
        assert "points" in row


def test_leaderboard_404_for_unknown_session(client: TestClient) -> None:
    r = client.get("/api/v1/sessions/999999/results")
    assert r.status_code == 404


def test_laps_happy_paginated(client: TestClient) -> None:
    r = client.get("/api/v1/sessions/1/laps?limit=10")
    assert r.status_code == 200
    body = r.json()
    assert len(body["data"]) == 10
    lap_nums = [l["lap_number"] for l in body["data"]]
    assert lap_nums == sorted(lap_nums)
    assert body["page"]["has_more"] is True
    assert body["page"]["next_cursor"]


def test_laps_filtered_by_driver(client: TestClient) -> None:
    r = client.get("/api/v1/sessions/1/laps?driver_id=1&limit=5")
    body = r.json()
    assert all(l["driver_id"] == 1 for l in body["data"])
    lap_nums = [l["lap_number"] for l in body["data"]]
    assert lap_nums == sorted(lap_nums)


def test_laps_filtered_by_lap_range(client: TestClient) -> None:
    r = client.get("/api/v1/sessions/1/laps?from_lap=10&to_lap=15&driver_id=1&limit=20")
    body = r.json()
    nums = [l["lap_number"] for l in body["data"]]
    assert nums == [10, 11, 12, 13, 14, 15]


def test_laps_cursor_walks_full_driver_history(client: TestClient) -> None:
    """Walk every lap for driver 1 across pages; no duplicates, no gaps."""
    seen: list[int] = []
    cursor: str | None = None
    for _ in range(50):
        url = "/api/v1/sessions/1/laps?driver_id=1&limit=20"
        if cursor:
            url += f"&cursor={cursor}"
        body = client.get(url).json()
        seen.extend(l["lap_number"] for l in body["data"])
        if not body["page"]["has_more"]:
            break
        cursor = body["page"]["next_cursor"]
    assert seen == sorted(seen), "no out-of-order laps across pages"
    assert len(seen) == len(set(seen)), "no duplicates across pages"


def test_laps_404_for_unknown_session(client: TestClient) -> None:
    r = client.get("/api/v1/sessions/999999/laps")
    assert r.status_code == 404


def test_laps_invalid_cursor(client: TestClient) -> None:
    r = client.get("/api/v1/sessions/1/laps?cursor=garbage!!!")
    assert r.status_code == 400
