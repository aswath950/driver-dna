"""Tests for /api/v1/events/{event_id}/sessions and /api/v1/sessions/{id}."""

from __future__ import annotations

import sqlalchemy as sa
from fastapi.testclient import TestClient

from app.core.config import settings


def _pick_event_with_sessions() -> int:
    """Find any event id in the seed that actually has child sessions."""
    eng = sa.create_engine(settings.DATABASE_URL_SYNC, future=True)
    with eng.connect() as conn:
        row = conn.execute(sa.text(
            "SELECT event_id FROM sessions GROUP BY event_id ORDER BY event_id LIMIT 1"
        )).first()
    assert row is not None, "seed has no sessions"
    return int(row[0])


def test_sessions_for_known_event(client: TestClient) -> None:
    event_id = _pick_event_with_sessions()
    r = client.get(f"/api/v1/events/{event_id}/sessions")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert len(body) > 0
    for s in body:
        assert s["event_id"] == event_id
        assert s["type"] in {"FP1", "FP2", "FP3", "Q", "SQ", "S", "R"}


def test_sessions_for_unknown_event_returns_404(client: TestClient) -> None:
    r = client.get("/api/v1/events/999999/sessions")
    assert r.status_code == 404
    assert r.json()["type"].endswith("/not_found")


def test_get_session_happy(client: TestClient) -> None:
    r = client.get("/api/v1/sessions/1")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == 1
    assert body["type"]


def test_get_session_404(client: TestClient) -> None:
    r = client.get("/api/v1/sessions/999999")
    assert r.status_code == 404
    body = r.json()
    assert body["type"].endswith("/not_found")
    assert "session" in body["detail"]
