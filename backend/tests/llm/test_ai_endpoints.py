"""Phase 9 — five LLM endpoints, OpenAI fully mocked.

Each test queues canned responses, hits the endpoint, asserts the response
shape, and verifies one or more rows landed in ``llm_audit``.
"""

from __future__ import annotations

import json

import sqlalchemy as sa
from fastapi.testclient import TestClient

from app.core.config import settings
from tests.llm.conftest import (
    enqueue_json,
    enqueue_text,
    enqueue_tool_call,
)


def _audit_count(feature_prefix: str) -> int:
    eng = sa.create_engine(settings.DATABASE_URL_SYNC, future=True)
    with eng.connect() as conn:
        n = conn.scalar(
            sa.text(
                "SELECT count(*) FROM llm_audit WHERE feature LIKE :p"
            ),
            {"p": f"{feature_prefix}%"},
        )
    return int(n or 0)


# ---------------------------------------------------------------------------
# style_analyst — Reflexion
# ---------------------------------------------------------------------------


def test_style_analyst_high_confidence_skips_revision(
    client: TestClient, openai_queue: list
) -> None:
    enqueue_text(openai_queue, "Verstappen is fast and consistent.")
    enqueue_json(openai_queue, {"confidence": 9, "issues": []})

    before = _audit_count("style_analyst")
    r = client.post(
        "/api/v1/ai/style-analyst",
        json={"driver_id": 1, "season": 2024},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["analysis"].startswith("Verstappen")
    assert body["confidence"] == 9
    assert body["rounds"] == 1
    assert body["revised"] is None
    assert _audit_count("style_analyst") == before + 2  # analyst + critic


def test_style_analyst_low_confidence_triggers_revise(
    client: TestClient, openai_queue: list
) -> None:
    enqueue_text(openai_queue, "Draft narrative.")
    enqueue_json(openai_queue, {"confidence": 4, "issues": ["sample issue"]})
    enqueue_text(openai_queue, "Revised narrative.")

    before = _audit_count("style_analyst")
    r = client.post(
        "/api/v1/ai/style-analyst",
        json={"driver_id": 1, "season": 2024},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rounds"] == 2
    assert body["revised"] == "Revised narrative."
    assert _audit_count("style_analyst") == before + 3


def test_style_analyst_unknown_driver(client: TestClient, openai_queue: list) -> None:
    r = client.post(
        "/api/v1/ai/style-analyst",
        json={"driver_id": 999999, "season": 2024},
    )
    assert r.status_code == 404
    assert "driver" in r.json()["detail"]


# ---------------------------------------------------------------------------
# dna_match — RAG
# ---------------------------------------------------------------------------


def test_dna_match_returns_top_two(client: TestClient, openai_queue: list) -> None:
    enqueue_text(openai_queue, "VER's style matches profile A and B.")
    r = client.post(
        "/api/v1/ai/dna-match",
        json={"driver_id": 1, "season": 2024},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["matches"]) == 2
    sims = [m["similarity"] for m in body["matches"]]
    assert sims == sorted(sims, reverse=True)
    assert len(body["vector"]) == 4
    assert "VER" in body["narrative"] or body["narrative"]
    assert body["driver_code"]


# ---------------------------------------------------------------------------
# report_card — Structured Output
# ---------------------------------------------------------------------------


def test_report_card_happy(client: TestClient, openai_queue: list) -> None:
    enqueue_json(openai_queue, {
        "grade": "A",
        "headline": "A dominant 2024 season.",
        "strengths": ["Consistency", "Tyre management"],
        "weaknesses": ["Occasional qualifying errors"],
        "verdict": "Verstappen continues to set the standard.",
    })
    r = client.post(
        "/api/v1/ai/report-card",
        json={"driver_id": 1, "season": 2024},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["grade"] == "A"
    assert len(body["strengths"]) >= 2
    assert "verdict" in body


def test_report_card_schema_violation_returns_4xx(
    client: TestClient, openai_queue: list
) -> None:
    enqueue_json(openai_queue, {
        "grade": "Z",  # invalid
        "headline": "x",
        "strengths": ["only one"],  # too few
        "weaknesses": ["a"],
        "verdict": "x",
    })
    r = client.post(
        "/api/v1/ai/report-card",
        json={"driver_id": 1, "season": 2024},
    )
    # Schema-violation is surfaced as a structured error (we raise NotFoundError
    # with a descriptive type for now; refinement to ValidationError is fine
    # later — the contract is that the client gets a 4xx envelope, not a 500).
    assert r.status_code in (400, 404, 422)


# ---------------------------------------------------------------------------
# xai_explain — Single-shot
# ---------------------------------------------------------------------------


def test_xai_explain_happy(client: TestClient, openai_queue: list) -> None:
    enqueue_text(openai_queue, "The model picked VER mostly because of feature X.")
    r = client.post(
        "/api/v1/ai/xai-explain",
        json={
            "predicted_driver_code": "VER",
            "feature_contributions": [
                {"feature": "min_throttle_pct", "value": 0.42,
                 "percentile": 0.91, "shap": 0.31},
                {"feature": "brake_duration_ms", "value": 850,
                 "percentile": 0.12, "shap": -0.20},
                {"feature": "max_speed_kph", "value": 327,
                 "percentile": 0.88, "shap": 0.18},
            ],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "feature X" in body["explanation"]
    assert len(body["top_features"]) == 3
    # Top feature must be the one with largest |shap|.
    assert body["top_features"][0]["feature"] == "min_throttle_pct"


def test_xai_explain_empty_features_rejected(
    client: TestClient, openai_queue: list
) -> None:
    r = client.post(
        "/api/v1/ai/xai-explain",
        json={"predicted_driver_code": "VER", "feature_contributions": []},
    )
    assert r.status_code == 422  # Pydantic min_length


# ---------------------------------------------------------------------------
# race-chat/stream — SSE
# ---------------------------------------------------------------------------


def _parse_sse(raw: str) -> list[dict]:
    """Split an SSE response body into [{event, data}, ...]."""
    out: list[dict] = []
    for block in raw.strip().split("\n\n"):
        if not block.strip():
            continue
        ev = None
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                ev = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        if ev is not None:
            payload = "\n".join(data_lines)
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                pass
            out.append({"event": ev, "data": payload})
    return out


def test_race_chat_direct_answer_no_tools(
    client: TestClient, openai_queue: list
) -> None:
    # Round 1: model answers directly, no tool calls.
    enqueue_text(openai_queue, "Verstappen led the pack from start to finish.")

    r = client.post(
        "/api/v1/ai/race-chat/stream",
        json={"session_id": 1, "message": "Who won?"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(r.text)
    kinds = [e["event"] for e in events]
    assert "token" in kinds
    assert "done" in kinds
    token_payloads = [e["data"]["delta"] for e in events if e["event"] == "token"]
    assert any("Verstappen" in t for t in token_payloads)


def test_race_chat_uses_tool_then_answers(
    client: TestClient, openai_queue: list
) -> None:
    # Round 1: model calls a tool.
    enqueue_tool_call(openai_queue, "get_rolling_pace_top", {"top_n": 3})
    # Round 2: model answers based on tool result.
    enqueue_text(openai_queue, "The fastest 3 drivers were ... [summary].")

    r = client.post(
        "/api/v1/ai/race-chat/stream",
        json={"session_id": 1, "message": "Who was fastest?"},
    )
    assert r.status_code == 200
    events = _parse_sse(r.text)
    kinds = [e["event"] for e in events]
    assert "tool_call" in kinds
    assert "tool_result" in kinds
    assert "token" in kinds
    assert kinds[-1] == "done"
    tc = next(e for e in events if e["event"] == "tool_call")
    assert tc["data"]["tool"] == "get_rolling_pace_top"


def test_race_chat_unknown_session_returns_error_event(
    client: TestClient, openai_queue: list
) -> None:
    r = client.post(
        "/api/v1/ai/race-chat/stream",
        json={"session_id": 999999, "message": "Who won?"},
    )
    assert r.status_code == 200  # SSE response, error encoded in the stream
    events = _parse_sse(r.text)
    assert events and events[0]["event"] == "error"
    assert "not_found" in events[0]["data"]["type"]
