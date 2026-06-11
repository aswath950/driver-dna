"""OpenAPI 3.1 conformance + error envelope reference checks."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_openapi_version_is_3_1(client: TestClient) -> None:
    r = client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    assert spec["openapi"].startswith("3.1"), f"got {spec['openapi']}"


def test_error_envelope_schema_present(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    schemas = spec["components"]["schemas"]
    assert "ErrorEnvelope" in schemas
    env = schemas["ErrorEnvelope"]
    required_props = {"type", "title", "status"}
    assert required_props.issubset(set(env["properties"].keys()))


def test_v1_endpoints_reference_error_envelope(client: TestClient) -> None:
    """Every endpoint mounted under /api/v1 should declare ErrorEnvelope in
    its 4xx/5xx responses (this is what the router-level ``responses=``
    in app/api/v1/__init__.py guarantees).
    """
    spec = client.get("/openapi.json").json()
    paths = spec["paths"]
    v1_paths = [p for p in paths if p.startswith("/api/v1")]
    assert v1_paths, "no v1 paths registered"
    for p in v1_paths:
        for method, op in paths[p].items():
            if method.startswith("x-"):
                continue
            responses = op.get("responses", {})
            error_codes = {code for code in responses if code.startswith(("4", "5"))}
            assert error_codes, f"{method.upper()} {p} declares no error responses"
            for code in error_codes:
                ref = (
                    responses[code]
                    .get("content", {})
                    .get("application/json", {})
                    .get("schema", {})
                    .get("$ref", "")
                )
                assert "ErrorEnvelope" in ref, (
                    f"{method.upper()} {p} {code} does not reference ErrorEnvelope "
                    f"(got: {ref!r})"
                )


def test_openapi_paths_include_v1_namespace(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    assert any(p.startswith("/api/v1/") for p in spec["paths"])
