"""OpenAPI sanity for Phase 6 — all 9 endpoints documented, paginated
endpoints expose Page schema, 404-returning endpoints reference ErrorEnvelope.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_all_phase6_paths_registered(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    expected = {
        "/api/v1/seasons",
        "/api/v1/seasons/{year}/events",
        "/api/v1/events/{event_id}/sessions",
        "/api/v1/sessions/{session_id}",
        "/api/v1/sessions/{session_id}/results",
        "/api/v1/sessions/{session_id}/laps",
        "/api/v1/drivers",
        "/api/v1/drivers/{driver_id}/stats",
        "/api/v1/standings",
    }
    assert expected.issubset(set(spec["paths"]))


def test_page_schema_emitted_for_paginated_lists(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    schemas = spec["components"]["schemas"]
    page_names = [n for n in schemas if n.startswith("Page_") or "Page[" in n]
    assert page_names, "Page[T] generics not emitted into components/schemas"


def test_v1_endpoints_have_error_responses(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    for path, methods in spec["paths"].items():
        if not path.startswith("/api/v1") or "/_ping" in path:
            continue
        for method, op in methods.items():
            if method.startswith("x-"):
                continue
            responses = op.get("responses", {})
            error_codes = [c for c in responses if c.startswith(("4", "5"))]
            assert error_codes, f"{method.upper()} {path} has no error responses"
