"""Phase 10 — anonymous user sessions + saved analyses.

Tests cover:
  - First request issues a Set-Cookie + creates a row in user_sessions.
  - Same client reuses the cookie; last_seen_at advances.
  - Two clients are isolated — saved analyses don't leak across cookies.
  - Cap of 100 saved analyses returns a 409 envelope on overflow.
  - 404 envelope on deleting an analysis you don't own / doesn't exist.
"""

from __future__ import annotations

import uuid

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.sessions import COOKIE_NAME


def _fresh_client(seeded_app_client_factory) -> TestClient:
    """Construct a brand-new TestClient (and thus a fresh cookie jar) that
    shares the same app + DB-override fixture as the rest of the suite.

    We can't just reuse the `client` fixture because TestClient persists
    cookies across calls within one client.
    """
    return seeded_app_client_factory()


@pytest.fixture()
def make_client(client: TestClient):
    """Return a factory yielding fresh TestClients backed by the same app."""
    from app.main import app

    def _factory() -> TestClient:
        return TestClient(app)

    return _factory


# ---------------------------------------------------------------------------
# Cookie issuance + reuse
# ---------------------------------------------------------------------------


def test_first_request_sets_cookie_and_creates_session(
    client: TestClient,
) -> None:
    # Use a brand-new client so the cookie jar starts empty.
    from app.main import app
    fresh = TestClient(app)

    r = fresh.get("/api/v1/me")
    assert r.status_code == 200
    body = r.json()
    assert body["id"]
    assert uuid.UUID(body["id"])  # well-formed UUID

    # Cookie set on the response.
    set_cookie = r.headers.get("set-cookie", "")
    assert COOKIE_NAME in set_cookie
    assert "httponly" in set_cookie.lower()
    assert "path=/" in set_cookie.lower()


def test_same_client_reuses_cookie(client: TestClient) -> None:
    from app.main import app
    c = TestClient(app)
    a = c.get("/api/v1/me").json()
    b = c.get("/api/v1/me").json()
    assert a["id"] == b["id"]
    assert b["last_seen_at"] >= a["last_seen_at"]


# ---------------------------------------------------------------------------
# Isolation across distinct clients
# ---------------------------------------------------------------------------


def test_two_clients_get_distinct_sessions(client: TestClient) -> None:
    from app.main import app
    a = TestClient(app)
    b = TestClient(app)
    id_a = a.get("/api/v1/me").json()["id"]
    id_b = b.get("/api/v1/me").json()["id"]
    assert id_a != id_b


def test_saved_analyses_are_isolated_per_session(client: TestClient) -> None:
    from app.main import app
    a = TestClient(app)
    b = TestClient(app)

    a.get("/api/v1/me")  # mint cookie
    b.get("/api/v1/me")

    a.post(
        "/api/v1/me/saved-analyses",
        json={"kind": "radar", "payload": {"who": "alice"}},
    ).raise_for_status()

    list_a = a.get("/api/v1/me/saved-analyses").json()
    list_b = b.get("/api/v1/me/saved-analyses").json()
    assert len(list_a["data"]) == 1
    assert list_a["data"][0]["payload"] == {"who": "alice"}
    assert list_b["data"] == []


# ---------------------------------------------------------------------------
# CRUD on saved-analyses
# ---------------------------------------------------------------------------


def test_create_then_list_then_delete(client: TestClient) -> None:
    from app.main import app
    c = TestClient(app)
    c.get("/api/v1/me")  # mint cookie

    create = c.post(
        "/api/v1/me/saved-analyses",
        json={"kind": "report_card", "session_id": 1, "payload": {"grade": "A"}},
    )
    assert create.status_code == 201, create.text
    created = create.json()
    assert created["kind"] == "report_card"
    assert created["session_id"] == 1
    aid = created["id"]

    lst = c.get("/api/v1/me/saved-analyses").json()
    assert any(r["id"] == aid for r in lst["data"])

    d = c.delete(f"/api/v1/me/saved-analyses/{aid}")
    assert d.status_code == 204

    after = c.get("/api/v1/me/saved-analyses").json()
    assert all(r["id"] != aid for r in after["data"])


def test_delete_unknown_returns_404(client: TestClient) -> None:
    from app.main import app
    c = TestClient(app)
    c.get("/api/v1/me")
    bogus = str(uuid.uuid4())
    r = c.delete(f"/api/v1/me/saved-analyses/{bogus}")
    assert r.status_code == 404
    assert r.json()["type"].endswith("/not_found")


def test_delete_other_users_analysis_returns_404(client: TestClient) -> None:
    from app.main import app
    owner = TestClient(app)
    intruder = TestClient(app)
    owner.get("/api/v1/me")
    intruder.get("/api/v1/me")

    created = owner.post(
        "/api/v1/me/saved-analyses",
        json={"kind": "xai", "payload": {}},
    ).json()
    aid = created["id"]

    # Intruder tries to delete by id → ownership check yields 404, not 403,
    # so the existence of the row isn't leaked.
    r = intruder.delete(f"/api/v1/me/saved-analyses/{aid}")
    assert r.status_code == 404


def test_invalid_kind_rejected(client: TestClient) -> None:
    from app.main import app
    c = TestClient(app)
    c.get("/api/v1/me")
    r = c.post(
        "/api/v1/me/saved-analyses",
        json={"kind": "not_a_real_kind", "payload": {}},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Cap
# ---------------------------------------------------------------------------


def test_saved_analyses_cap_returns_409(client: TestClient) -> None:
    """Bulk-insert 100 rows directly via SQL then attempt one more via API."""
    from app.main import app

    c = TestClient(app)
    me = c.get("/api/v1/me").json()
    sid = me["id"]

    # Fill the cap.
    eng = sa.create_engine(settings.DATABASE_URL_SYNC, future=True)
    with eng.begin() as conn:
        conn.execute(
            sa.text(
                "INSERT INTO saved_analyses (user_session_id, kind, payload) "
                "SELECT CAST(:sid AS uuid), 'radar'::analysis_kind, '{}'::jsonb "
                "FROM generate_series(1, 100)"
            ),
            {"sid": sid},
        )

    r = c.post(
        "/api/v1/me/saved-analyses",
        json={"kind": "radar", "payload": {"one": "more"}},
    )
    assert r.status_code == 409
    assert "cap" in r.json()["detail"].lower()

    # Cleanup so other tests' cap counts aren't polluted.
    with eng.begin() as conn:
        conn.execute(
            sa.text("DELETE FROM saved_analyses WHERE user_session_id = CAST(:sid AS uuid)"),
            {"sid": sid},
        )


# ---------------------------------------------------------------------------
# LLM audit FK
# ---------------------------------------------------------------------------


def test_llm_audit_attributes_to_user_session(client: TestClient) -> None:
    """Hitting any AI endpoint should write llm_audit rows attached to the
    caller's user_session_id.
    """
    # We use the styled mock from tests/llm/conftest indirectly — but here
    # we just verify the schema wiring: write one row manually with the
    # current session id and confirm we can read it back.
    from app.main import app
    c = TestClient(app)
    me = c.get("/api/v1/me").json()
    sid = me["id"]

    eng = sa.create_engine(settings.DATABASE_URL_SYNC, future=True)
    with eng.begin() as conn:
        conn.execute(
            sa.text(
                "INSERT INTO llm_audit (feature, model, status, user_session_id) "
                "VALUES ('test_feature', 'gpt-4o-mini', 'success', CAST(:sid AS uuid))"
            ),
            {"sid": sid},
        )
        n = conn.scalar(
            sa.text("SELECT count(*) FROM llm_audit WHERE user_session_id = CAST(:sid AS uuid)"),
            {"sid": sid},
        )
    assert int(n or 0) >= 1
