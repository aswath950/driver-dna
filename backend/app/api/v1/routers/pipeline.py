"""Pipeline management endpoints — stats, hydrate (SSE), train (SSE), model metrics."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import AsyncIterator

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select

from app.core.config import settings
from app.core.deps import DB
from app.db.models import LapTime, Session, SessionDriver

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

# backend/ directory — cwd for `python -m app.etl ...`
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
# project root — cwd for model training (one above backend/)
_PROJECT_ROOT = _BACKEND_DIR.parent


@router.get("/stats", summary="Dataset and training stats from the DB.")
async def pipeline_stats(db: DB) -> dict:
    dataset_rows = (
        await db.execute(select(func.count()).select_from(LapTime))
    ).scalar_one()

    drivers = (
        await db.execute(
            select(func.count(SessionDriver.driver_id.distinct())).select_from(SessionDriver)
        )
    ).scalar_one()

    laps = (
        await db.execute(
            select(func.count(LapTime.session_id.distinct())).select_from(LapTime)
        )
    ).scalar_one()

    last_updated_row = (
        await db.execute(
            select(func.max(Session.date_start)).join(LapTime, LapTime.session_id == Session.id)
        )
    ).scalar_one_or_none()

    last_updated = last_updated_row.isoformat() if last_updated_row else None

    return {
        "dataset_rows": dataset_rows,
        "drivers": drivers,
        "laps": laps,
        "last_updated": last_updated,
    }


async def _stream_subprocess(args: list[str], cwd: Path) -> AsyncIterator[str]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(cwd),
    )
    assert proc.stdout is not None
    async for line in proc.stdout:
        yield f"data: {line.decode().rstrip()}\n\n"
    exit_code = await proc.wait()
    yield f"data: [done] exit_code={exit_code}\n\n"


@router.get("/model-metrics", summary="Latest model accuracy metrics from models/metrics.json.")
async def model_metrics() -> dict:
    metrics_path = _PROJECT_ROOT / "models" / "metrics.json"
    if not metrics_path.exists():
        return {"cv_accuracy": None, "train_accuracy": None}
    with open(metrics_path) as f:
        data = json.load(f)
    return {
        "cv_accuracy": data.get("cv_mean") or data.get("cv_accuracy"),
        "train_accuracy": data.get("train_accuracy"),
    }


@router.get("/gp-schedule", summary="List Grand Prix names for a year from OpenF1.")
async def gp_schedule(year: int = Query(..., description="Season year, e.g. 2024.")) -> list[dict]:
    url = f"{settings.OPENF1_BASE_URL}/meetings?year={year}"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        meetings = resp.json()
    races = [
        {"name": m["meeting_name"], "date": m.get("date_start", "")[:10]}
        for m in meetings
        if not m.get("is_cancelled") and "testing" not in m.get("meeting_name", "").lower()
    ]
    races.sort(key=lambda m: m["date"])
    for i, r in enumerate(races, start=1):
        r["round"] = i
    return races


@router.get("/hydrate", summary="Hydrate one session from OpenF1 (SSE stream).")
async def hydrate(
    year: int = Query(..., description="Season year, e.g. 2024."),
    gp: str = Query(..., description="Grand Prix name, e.g. 'Italian Grand Prix'."),
    session: str | None = Query(None, description="Session type (R/Q/FP1/FP2/FP3). Omit to hydrate all sessions in the weekend."),
) -> StreamingResponse:
    args = [
        sys.executable, "-m", "app.etl",
        "hydrate",
        "--year", str(year),
        "--gp", gp,
    ]
    if session:
        args += ["--session", session]
    return StreamingResponse(
        _stream_subprocess(args, _BACKEND_DIR),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/fetch-telemetry", summary="Download and cache all car telemetry for a session (SSE stream).")
async def fetch_telemetry(
    session_id: int = Query(..., description="Database session ID, e.g. 73."),
) -> StreamingResponse:
    args = [
        sys.executable, "-m", "app.etl",
        "fetch-telemetry",
        "--session-id", str(session_id),
    ]
    return StreamingResponse(
        _stream_subprocess(args, _BACKEND_DIR),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/telemetry-status", summary="Check whether a session's telemetry is cached.")
async def telemetry_status(
    db: DB,
    session_id: int = Query(..., description="Database session ID."),
) -> dict:
    session = await db.get(Session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")
    fetched_at = session.telemetry_fetched_at
    return {
        "session_id": session_id,
        "fetched_at": fetched_at.isoformat() if fetched_at else None,
    }


@router.get("/train", summary="Run model training (SSE stream).")
async def train() -> StreamingResponse:
    venv_python = _PROJECT_ROOT / ".venv" / "bin" / "python"
    python_exec = str(venv_python) if venv_python.exists() else sys.executable
    model_script = _PROJECT_ROOT / "src" / "model.py"
    return StreamingResponse(
        _stream_subprocess([python_exec, str(model_script)], _PROJECT_ROOT),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
