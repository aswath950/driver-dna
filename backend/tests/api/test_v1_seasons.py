"""Tests for /api/v1/seasons and /api/v1/seasons/{year}/events."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_list_seasons_default(client: TestClient) -> None:
    r = client.get("/api/v1/seasons")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"data", "page"}
    years = [s["year"] for s in body["data"]]
    assert years == sorted(years, reverse=True), "should be newest-first"
    assert 2024 in years and 2015 in years


def test_seasons_cursor_paginates(client: TestClient) -> None:
    seen: list[int] = []
    cursor: str | None = None
    for _ in range(20):
        url = "/api/v1/seasons?limit=3"
        if cursor:
            url += f"&cursor={cursor}"
        body = client.get(url).json()
        seen.extend(s["year"] for s in body["data"])
        if not body["page"]["has_more"]:
            break
        cursor = body["page"]["next_cursor"]
    assert sorted(seen, reverse=True) == seen
    assert set(seen) >= set(range(2015, 2025))


def test_seasons_limit_validation(client: TestClient) -> None:
    assert client.get("/api/v1/seasons?limit=0").status_code == 422
    assert client.get("/api/v1/seasons?limit=999").status_code == 422


def test_seasons_invalid_cursor(client: TestClient) -> None:
    r = client.get("/api/v1/seasons?cursor=garbage!!!")
    assert r.status_code == 400
    assert r.json()["type"].endswith("/bad_request")


def test_events_for_known_season(client: TestClient) -> None:
    r = client.get("/api/v1/seasons/2024/events?limit=5")
    assert r.status_code == 200
    body = r.json()
    rounds = [e["round"] for e in body["data"]]
    assert rounds == sorted(rounds), "events ordered by round ASC"
    assert all(e["season_id"] for e in body["data"])


def test_events_for_unknown_season_returns_404(client: TestClient) -> None:
    r = client.get("/api/v1/seasons/1999/events")
    assert r.status_code == 404
    body = r.json()
    assert body["type"].endswith("/not_found")
    assert "season" in body["detail"]
