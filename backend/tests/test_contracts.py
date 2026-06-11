"""Phase 4 contract tests.

Covers the three things every v1 endpoint inherits:
1. RFC 7807 error envelope shape on any error response.
2. Cursor pagination round-trip + invalid-cursor handling.
3. `X-Request-ID` and `API-Version` header behaviour.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.pagination import decode_cursor, encode_cursor

# ---------------------------------------------------------------------------
# Error envelope
# ---------------------------------------------------------------------------


def test_not_found_envelope_shape(client: TestClient) -> None:
    r = client.get("/api/v1/_ping/raise/not_found")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/problem+json")
    body = r.json()
    assert body["type"] == "https://driver-dna.dev/errors/not_found"
    assert body["title"] == "Resource not found"
    assert body["status"] == 404
    assert body["instance"] == "/api/v1/_ping/raise/not_found"
    assert body["request_id"]  # populated by middleware
    assert "ping" in body["detail"]


def test_bad_request_envelope(client: TestClient) -> None:
    r = client.get("/api/v1/_ping/raise/bad_request")
    assert r.status_code == 400
    body = r.json()
    assert body["type"].endswith("/bad_request")
    assert body["status"] == 400


def test_unknown_path_uses_envelope(client: TestClient) -> None:
    """FastAPI's default 404 must flow through the envelope handler."""
    r = client.get("/api/v1/does-not-exist")
    assert r.status_code == 404
    body = r.json()
    assert body["type"].endswith("/not_found")
    assert body["instance"] == "/api/v1/does-not-exist"


def test_validation_error_uses_envelope(client: TestClient) -> None:
    r = client.get("/api/v1/_ping/echo-limit?limit=9999")
    assert r.status_code == 422
    body = r.json()
    assert body["type"].endswith("/validation_error")
    assert body["status"] == 422


# ---------------------------------------------------------------------------
# Cursor pagination
# ---------------------------------------------------------------------------


def test_cursor_encode_decode_round_trip(client: TestClient) -> None:
    encoded = encode_cursor(42, 7)
    assert isinstance(encoded, str)
    assert "=" not in encoded  # padding stripped
    k, pk = decode_cursor(encoded)  # type: ignore[misc]
    assert k == 42
    assert pk == 7


def test_cursor_decode_none(client: TestClient) -> None:
    assert decode_cursor(None) is None
    assert decode_cursor("") is None


def test_cursor_decode_garbage_raises_400(client: TestClient) -> None:
    r = client.get("/api/v1/_ping/page?cursor=not-a-real-cursor!!!")
    assert r.status_code == 400
    body = r.json()
    assert body["type"].endswith("/bad_request")
    assert "invalid cursor" in (body.get("detail") or "")


def test_cursor_paginates_full_universe(client: TestClient) -> None:
    """Walk all 100 integers via repeated cursor calls."""
    seen: list[int] = []
    cursor: str | None = None
    for _ in range(40):  # safety bound
        url = "/api/v1/_ping/page?limit=10"
        if cursor:
            url += f"&cursor={cursor}"
        r = client.get(url)
        assert r.status_code == 200
        body = r.json()
        seen.extend(body["data"])
        assert body["page"]["limit"] == 10
        if not body["page"]["has_more"]:
            assert body["page"]["next_cursor"] is None
            break
        cursor = body["page"]["next_cursor"]
        assert cursor
    assert seen == list(range(1, 101))


def test_page_envelope_keys(client: TestClient) -> None:
    r = client.get("/api/v1/_ping/page?limit=3")
    body = r.json()
    assert set(body.keys()) == {"data", "page"}
    assert set(body["page"].keys()) >= {"limit", "has_more"}


# ---------------------------------------------------------------------------
# Headers: request-id + api-version
# ---------------------------------------------------------------------------


def test_request_id_minted_when_absent(client: TestClient) -> None:
    r = client.get("/api/v1/_ping")
    assert "X-Request-ID" in r.headers
    assert len(r.headers["X-Request-ID"]) >= 16


def test_request_id_echoed_when_client_supplies(client: TestClient) -> None:
    r = client.get("/api/v1/_ping", headers={"X-Request-ID": "client-supplied-123"})
    assert r.headers["X-Request-ID"] == "client-supplied-123"


def test_api_version_header_on_v1_routes(client: TestClient) -> None:
    r = client.get("/api/v1/_ping")
    assert r.headers.get("API-Version") == "1"


def test_api_version_header_absent_on_non_v1_routes(client: TestClient) -> None:
    r = client.get("/healthz")
    assert "API-Version" not in r.headers


def test_request_id_propagates_into_error_envelope(client: TestClient) -> None:
    r = client.get(
        "/api/v1/_ping/raise/not_found",
        headers={"X-Request-ID": "test-rid-abc"},
    )
    assert r.headers["X-Request-ID"] == "test-rid-abc"
    assert r.json()["request_id"] == "test-rid-abc"
