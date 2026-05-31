"""Phase 7 — compute endpoints.

Five routes, all under ``/api/v1/sessions/{session_id}/``:

- analytics/rolling-pace
- analytics/gap-to-leader
- analytics/undercuts
- analytics/tyre-degradation
- compare

The first four reuse ``src.race_engine.RaceAnalyser`` via the analytics
service adapter. The fifth fetches fastest-lap telemetry directly from
OpenF1 (the DB has no telemetry traces).
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.v1.schemas.analytics import (
    Channel,
    ComparePayload,
    DegradationRow,
    GapRow,
    RollingPaceRow,
    UndercutEvent,
)
from app.core.deps import DB
from app.core.errors import ErrorEnvelope, NotFoundError
from app.db.repositories import sessions as sessions_repo
from app.services import analytics_service, compare_service

router = APIRouter(tags=["analytics"])

_NOT_FOUND = {404: {"model": ErrorEnvelope}}
_NOT_READY = {503: {"model": ErrorEnvelope}}


async def _ensure_session(db, session_id: int) -> None:
    if await sessions_repo.get_session(db, session_id) is None:
        raise NotFoundError("session", session_id)


@router.get(
    "/sessions/{session_id}/analytics/rolling-pace",
    response_model=list[RollingPaceRow],
    responses={**_NOT_FOUND, **_NOT_READY},
    summary="Rolling-average lap pace per driver.",
)
async def rolling_pace(
    session_id: int,
    db: DB,
    window: int = Query(default=5, ge=1, le=50, description="Rolling-window size in laps."),
) -> list[RollingPaceRow]:
    await _ensure_session(db, session_id)
    drv_map = await analytics_service._load_car_number_map(db, session_id)
    analyser = await analytics_service.build_analyser(db, session_id)
    rows = analytics_service.rolling_pace_rows(analyser, window=window, drv_map=drv_map)
    return [RollingPaceRow.model_validate(r) for r in rows]


@router.get(
    "/sessions/{session_id}/analytics/gap-to-leader",
    response_model=list[GapRow],
    responses={**_NOT_FOUND, **_NOT_READY},
    summary="Cumulative gap to the race leader per lap, per driver.",
)
async def gap_to_leader(session_id: int, db: DB) -> list[GapRow]:
    await _ensure_session(db, session_id)
    drv_map = await analytics_service._load_car_number_map(db, session_id)
    analyser = await analytics_service.build_analyser(db, session_id)
    rows = analytics_service.gap_to_leader_rows(analyser, drv_map=drv_map)
    return [GapRow.model_validate(r) for r in rows]


@router.get(
    "/sessions/{session_id}/analytics/undercuts",
    response_model=list[UndercutEvent],
    responses={**_NOT_FOUND, **_NOT_READY},
    summary="Detected undercut / overcut windows around pit stops.",
)
async def undercuts(session_id: int, db: DB) -> list[UndercutEvent]:
    await _ensure_session(db, session_id)
    drv_map = await analytics_service._load_car_number_map(db, session_id)
    analyser = await analytics_service.build_analyser(db, session_id)
    rows = analytics_service.undercut_events(analyser, drv_map=drv_map)
    return [UndercutEvent.model_validate(r) for r in rows]


@router.get(
    "/sessions/{session_id}/analytics/tyre-degradation",
    response_model=list[DegradationRow],
    responses={**_NOT_FOUND, **_NOT_READY},
    summary="Per-stint tyre degradation (sec/lap) via linear regression.",
)
async def tyre_degradation(session_id: int, db: DB) -> list[DegradationRow]:
    await _ensure_session(db, session_id)
    drv_map = await analytics_service._load_car_number_map(db, session_id)
    analyser = await analytics_service.build_analyser(db, session_id)
    rows = analytics_service.degradation_rows(analyser, drv_map=drv_map)
    return [DegradationRow.model_validate(r) for r in rows]


@router.get(
    "/sessions/{session_id}/compare",
    response_model=ComparePayload,
    responses={**_NOT_FOUND, **_NOT_READY, 400: {"model": ErrorEnvelope}},
    summary="Compare two drivers' fastest-lap telemetry (Speed/Throttle/Brake).",
)
async def compare(
    session_id: int,
    db: DB,
    driver_a: int = Query(..., description="Driver A internal id."),
    driver_b: int = Query(..., description="Driver B internal id."),
    channel: Channel = Query(
        default="Speed",
        description="Channel to compare. v1 supports Speed/Throttle/Brake; "
                    "TimeDelta + TrackMap are deferred to Phase 11+.",
    ),
) -> ComparePayload:
    payload = await compare_service.build_compare_payload(
        db,
        session_id=session_id,
        driver_a_id=driver_a,
        driver_b_id=driver_b,
        channel=channel,
    )
    return ComparePayload.model_validate(payload)
