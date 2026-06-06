"""Phase 7 — compute endpoints.

Routes under ``/api/v1/sessions/{session_id}/``:

- analytics/rolling-pace
- analytics/gap-to-leader
- analytics/undercuts
- analytics/tyre-degradation
- compare                 (Speed / Throttle / Brake / RPM / nGear / DRS / TimeDelta)
- compare/sectors         (SectorTimes — DB-backed)
- compare/track-map       (x/y position — OpenF1-backed)
- sector-times            (legacy alias kept for backwards compat)
"""

from __future__ import annotations

from typing import Literal

import plotly.graph_objects as go
from fastapi import APIRouter, Query
from sqlalchemy import select

from app.api.v1.schemas.analytics import (
    Channel,
    ComparePayload,
    CornerPerformancePayload,
    DegradationRow,
    GapRow,
    RollingPaceRow,
    SectorDriverSplits,
    SectorTimesPayload,
    TrackMapPayload,
    UndercutEvent,
)
from app.core.deps import DB
from app.core.errors import ErrorEnvelope, NotFoundError, UpstreamError
from app.db.models import Driver, SessionDriver
from app.db.repositories import laps as laps_repo
from app.db.repositories import sessions as sessions_repo
from app.services import analytics_service, compare_service, corner_service

router = APIRouter(tags=["analytics"])

_NOT_FOUND = {404: {"model": ErrorEnvelope}}
_NOT_READY = {503: {"model": ErrorEnvelope}}

# Channels served by the /compare endpoint (all return ComparePayload).
# SectorTimes and TrackMap have their own dedicated endpoints.
_TelemetryChannel = Literal[
    "Speed", "Throttle", "Brake", "RPM", "nGear", "DRS", "TimeDelta"
]


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


# ---------------------------------------------------------------------------
# Telemetry compare — car-data channels + TimeDelta
# ---------------------------------------------------------------------------


@router.get(
    "/sessions/{session_id}/compare",
    response_model=ComparePayload,
    responses={**_NOT_FOUND, **_NOT_READY, 400: {"model": ErrorEnvelope}},
    summary="Compare two drivers' fastest-lap telemetry (Speed/Throttle/Brake/RPM/nGear/DRS/TimeDelta).",
)
async def compare(
    session_id: int,
    db: DB,
    driver_a: int = Query(..., description="Driver A internal id."),
    driver_b: int = Query(..., description="Driver B internal id."),
    channel: _TelemetryChannel = Query(
        default="Speed",
        description=(
            "Telemetry channel. Car-data channels: Speed, Throttle, Brake, RPM, nGear, DRS. "
            "TimeDelta derives cumulative time difference from Speed traces. "
            "For SectorTimes use /compare/sectors; for TrackMap use /compare/track-map."
        ),
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


# ---------------------------------------------------------------------------
# Telemetry compare — SectorTimes (DB-backed)
# ---------------------------------------------------------------------------


@router.get(
    "/sessions/{session_id}/compare/sectors",
    response_model=SectorTimesPayload,
    responses={**_NOT_FOUND, **_NOT_READY, 400: {"model": ErrorEnvelope}},
    summary="Compare two drivers' fastest-lap sector times (S1/S2/S3) from the DB.",
)
async def compare_sectors(
    session_id: int,
    db: DB,
    driver_a: int = Query(..., description="Driver A internal id."),
    driver_b: int = Query(..., description="Driver B internal id."),
) -> SectorTimesPayload:
    await _ensure_session(db, session_id)
    payload = await compare_service.build_sector_times_payload(
        db,
        session_id=session_id,
        driver_a_id=driver_a,
        driver_b_id=driver_b,
    )
    return SectorTimesPayload.model_validate(payload)


# ---------------------------------------------------------------------------
# Telemetry compare — TrackMap (OpenF1 position)
# ---------------------------------------------------------------------------


@router.get(
    "/sessions/{session_id}/compare/track-map",
    response_model=TrackMapPayload,
    responses={**_NOT_FOUND, **_NOT_READY, 400: {"model": ErrorEnvelope}},
    summary="Compare two drivers' fastest-lap track positions (x/y).",
)
async def compare_track_map(
    session_id: int,
    db: DB,
    driver_a: int = Query(..., description="Driver A internal id."),
    driver_b: int = Query(..., description="Driver B internal id."),
) -> TrackMapPayload:
    payload = await compare_service.build_track_map_payload(
        db,
        session_id=session_id,
        driver_a_id=driver_a,
        driver_b_id=driver_b,
    )
    return TrackMapPayload.model_validate(payload)


# ---------------------------------------------------------------------------
# Legacy: /sector-times (kept for backwards compatibility with existing clients)
# ---------------------------------------------------------------------------


async def _resolve_driver_code(db, session_id: int, driver_id: int) -> str:
    row = (
        await db.execute(
            select(Driver.code)
            .join(SessionDriver, Driver.id == SessionDriver.driver_id)
            .where(
                SessionDriver.session_id == session_id,
                SessionDriver.driver_id == driver_id,
            )
        )
    ).first()
    if row is None:
        raise NotFoundError(
            "session_driver", f"session={session_id} driver={driver_id}"
        )
    return str(row.code)


@router.get(
    "/sessions/{session_id}/sector-times",
    response_model=SectorTimesPayload,
    responses={**_NOT_FOUND, **_NOT_READY},
    summary="Compare two drivers' fastest-lap sector times (S1/S2/S3).",
)
async def sector_times(
    session_id: int,
    db: DB,
    driver_a: int = Query(..., description="Driver A internal id."),
    driver_b: int = Query(..., description="Driver B internal id."),
) -> SectorTimesPayload:
    await _ensure_session(db, session_id)
    code_a = await _resolve_driver_code(db, session_id, driver_a)
    code_b = await _resolve_driver_code(db, session_id, driver_b)

    lap_a = await laps_repo.fastest_lap_sectors(db, session_id=session_id, driver_id=driver_a)
    lap_b = await laps_repo.fastest_lap_sectors(db, session_id=session_id, driver_id=driver_b)

    if lap_a is None:
        raise UpstreamError(f"no qualifying lap for driver={driver_a} session={session_id}")
    if lap_b is None:
        raise UpstreamError(f"no qualifying lap for driver={driver_b} session={session_id}")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=code_a,
        x=["S1", "S2", "S3"],
        y=[lap_a.sector1_ms, lap_a.sector2_ms, lap_a.sector3_ms],
    ))
    fig.add_trace(go.Bar(
        name=code_b,
        x=["S1", "S2", "S3"],
        y=[lap_b.sector1_ms, lap_b.sector2_ms, lap_b.sector3_ms],
    ))
    fig.update_layout(
        title=f"Sector Times: {code_a} vs {code_b} (fastest lap)",
        xaxis_title="Sector",
        yaxis_title="Time (ms)",
        barmode="group",
        template="plotly_dark",
        margin={"l": 60, "r": 30, "t": 60, "b": 50},
    )

    splits_a = SectorDriverSplits(
        driver_id=driver_a,
        code=code_a,
        lap_number=lap_a.lap_number,
        lap_time_ms=lap_a.lap_time_ms,
        sector1_ms=lap_a.sector1_ms,
        sector2_ms=lap_a.sector2_ms,
        sector3_ms=lap_a.sector3_ms,
    )
    splits_b = SectorDriverSplits(
        driver_id=driver_b,
        code=code_b,
        lap_number=lap_b.lap_number,
        lap_time_ms=lap_b.lap_time_ms,
        sector1_ms=lap_b.sector1_ms,
        sector2_ms=lap_b.sector2_ms,
        sector3_ms=lap_b.sector3_ms,
    )
    return SectorTimesPayload(
        session_id=session_id,
        driver_a=splits_a,
        driver_b=splits_b,
        figure_json=fig.to_json(),
    )


# ---------------------------------------------------------------------------
# Corner performance — team-vs-team comparison across corner types
# ---------------------------------------------------------------------------


@router.get(
    "/sessions/{session_id}/corner-performance",
    response_model=CornerPerformancePayload,
    responses={**_NOT_FOUND, **_NOT_READY, 400: {"model": ErrorEnvelope}},
    summary="Team-vs-team corner performance broken down by slow / medium / high speed corners.",
)
async def corner_performance(
    session_id: int,
    db: DB,
    team_a: int = Query(..., description="Team A internal id."),
    team_b: int = Query(..., description="Team B internal id."),
) -> CornerPerformancePayload:
    payload = await corner_service.build_corner_performance_payload(
        db,
        session_id=session_id,
        team_a_id=team_a,
        team_b_id=team_b,
    )
    return CornerPerformancePayload.model_validate(payload)
