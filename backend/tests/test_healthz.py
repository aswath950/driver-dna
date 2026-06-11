from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_healthz_ok(client: TestClient) -> None:
    client = TestClient(app)
    res = client.get("/healthz")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "env" in body
    assert "version" in body


def test_root_links(client: TestClient) -> None:
    client = TestClient(app)
    res = client.get("/")
    assert res.status_code == 200
    body = res.json()
    assert body["health"] == "/healthz"
    assert body["docs"] == "/docs"


def test_shared_src_on_syspath(client: TestClient) -> None:
    """Phase 1 wiring guarantee: importing app.main puts the repo root on
    sys.path so `from src.* import ...` resolves. We don't import a module
    here because the heavy src/ modules pull in numpy/pandas/xgboost that
    are intentionally NOT in the backend venv (they belong to the legacy
    Streamlit venv at the repo root). Phase 5 (ETL) pulls them in.
    """
    import sys
    from pathlib import Path

    import app.main  # noqa: F401 — side effect: sys.path insert

    repo_root = Path(app.main.__file__).resolve().parents[2]
    assert str(repo_root) in sys.path
    assert (repo_root / "src" / "race_engine.py").is_file()
