"""Tests for /api/v1/drivers, /api/v1/drivers/{id}/stats, /api/v1/standings."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_list_drivers_default(client: TestClient) -> None:
    r = client.get("/api/v1/drivers?limit=5")
    assert r.status_code == 200
    body = r.json()
    codes = [d["code"] for d in body["data"]]
    assert codes == sorted(codes), "drivers ordered by code ASC"
    for d in body["data"]:
        assert len(d["code"]) >= 2


def test_list_drivers_filtered_by_season(client: TestClient) -> None:
    r = client.get("/api/v1/drivers?season=2024&limit=50")
    assert r.status_code == 200
    body = r.json()
    assert len(body["data"]) > 0
    # All seeded drivers raced in 2024, so this should still return rows.


def test_list_drivers_filtered_by_team(client: TestClient) -> None:
    r = client.get("/api/v1/drivers?team=Ferrari&limit=10")
    body = r.json()
    assert all(
        d.get("current_team") is None
        or d["current_team"]["name"] == "Ferrari"
        for d in body["data"]
    )


def test_drivers_cursor_paginates(client: TestClient) -> None:
    seen: list[str] = []
    cursor: str | None = None
    for _ in range(15):
        url = "/api/v1/drivers?limit=4"
        if cursor:
            url += f"&cursor={cursor}"
        body = client.get(url).json()
        seen.extend(d["code"] for d in body["data"])
        if not body["page"]["has_more"]:
            break
        cursor = body["page"]["next_cursor"]
    assert seen == sorted(seen)
    assert len(seen) == len(set(seen))


def test_driver_stats_happy(client: TestClient) -> None:
    r = client.get("/api/v1/drivers/1/stats?season=2024")
    assert r.status_code == 200
    body = r.json()
    assert body["driver_id"] == 1
    assert isinstance(body["points"], str)  # Decimal serialised as string
    assert body["wins"] >= 0


def test_driver_stats_missing_season_param(client: TestClient) -> None:
    r = client.get("/api/v1/drivers/1/stats")
    assert r.status_code == 422


def test_driver_stats_unknown_driver(client: TestClient) -> None:
    r = client.get("/api/v1/drivers/999999/stats?season=2024")
    assert r.status_code == 404


def test_driver_stats_no_data_for_season(client: TestClient) -> None:
    r = client.get("/api/v1/drivers/1/stats?season=1980")
    assert r.status_code == 404


def test_standings_happy(client: TestClient) -> None:
    r = client.get("/api/v1/standings?season=2024")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert len(body) > 0
    points = [float(row["points"]) for row in body]
    assert points == sorted(points, reverse=True), "ordered by points DESC"
    positions = [row["position"] for row in body]
    assert positions == list(range(1, len(positions) + 1))
    assert body[0]["driver"]["code"]


def test_standings_missing_season_param(client: TestClient) -> None:
    r = client.get("/api/v1/standings")
    assert r.status_code == 422


def test_standings_empty_season(client: TestClient) -> None:
    r = client.get("/api/v1/standings?season=1980")
    assert r.status_code == 200
    assert r.json() == []
