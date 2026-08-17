"""Orchestration for corner performance: loads circuit + telemetry from DB,
delegates computation to corner_compute, returns a serializable payload.

Mirrors the structure of compare_service.py:
- Cache-first telemetry loading via _fetch_fastest_lap_data
- Same error types (NotFoundError, UpstreamError, BadRequestError)
- Returns a plain dict that the router validates against CornerPerformancePayload
"""

from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src.openf1 import OpenF1Client

from app.core.errors import BadRequestError, NotFoundError, UpstreamError
from app.db.models import Driver, SessionDriver, Team
from app.db.repositories import circuits as circuits_repo
from app.services import corner_compute as cc
from app.services.compare_service import (
    _fetch_fastest_lap_data,
    _resolve_session_key,
)

logger = logging.getLogger(__name__)


async def _load_team_drivers(
    db: AsyncSession,
    session_id: int,
    team_id: int,
) -> list[tuple[int, int, str]]:
    """Return list of (driver_id, car_number, driver_code) for a team in a session."""
    rows = (
        await db.execute(
            select(SessionDriver.driver_id, SessionDriver.car_number, Driver.code)
            .join(Driver, Driver.id == SessionDriver.driver_id)
            .where(
                SessionDriver.session_id == session_id,
                SessionDriver.team_id == team_id,
            )
        )
    ).all()
    return [(int(r.driver_id), int(r.car_number), str(r.code)) for r in rows]


async def _load_team_info(db: AsyncSession, team_id: int) -> dict:
    """Return {id, name, color_hex} for a team, raising NotFoundError if absent."""
    team = await db.get(Team, team_id)
    if team is None:
        raise NotFoundError("team", team_id)
    return {
        "id": team.id,
        "name": team.name,
        "color_hex": team.color_hex or "#888888",
    }


async def _fetch_driver_data(
    db: AsyncSession,
    session_id: int,
    driver_id: int,
    car_number: int,
    client: OpenF1Client,
    session_key: int,
    laps_df: pd.DataFrame | None,
) -> dict | None:
    """Thin wrapper: cache-first telemetry fetch returning the N_POINTS-pt processed dict."""
    return await _fetch_fastest_lap_data(
        db,
        session_id=session_id,
        driver_id=driver_id,
        client=client,
        session_key=session_key,
        driver_number=car_number,
        laps_df=laps_df,
    )


async def build_corner_performance_payload(
    db: AsyncSession,
    *,
    session_id: int,
    team_a_id: int,
    team_b_id: int,
) -> dict:
    """Build the full corner performance payload for two teams.

    Steps:
    1. Validate inputs.
    2. Load circuit geometry (x, y required for corner detection).
    3. Load team info and driver lists for each team.
    4. Fetch fastest-lap telemetry for each driver (cache-first).
    5. Detect corners from telemetry speed minima; classify by apex speed.
    6. Compute per-driver metrics; median-aggregate per team.
    7. Build class summary and three Plotly figures.
    8. Return serializable dict.

    Raises:
        BadRequestError:  team_a == team_b
        NotFoundError:    team not found in DB or not participating in session
        UpstreamError:    missing circuit geometry, or no usable telemetry for a team
    """
    if team_a_id == team_b_id:
        raise BadRequestError("team_a and team_b must differ")

    # ── Session + circuit ────────────────────────────────────────────────────
    session_key = await _resolve_session_key(db, session_id)

    circuit = await circuits_repo.get_for_session(db, session_id)
    if circuit is None:
        raise UpstreamError(
            f"no circuit linked to session {session_id} — was the event hydrated?"
        )
    if not circuit.x or not circuit.y:
        raise UpstreamError(
            f"circuit {circuit.name!r} has no outline geometry — "
            "run `python -m app.etl seed-circuits` to seed circuit x/y"
        )

    # ── Team info + drivers ──────────────────────────────────────────────────
    team_a_info = await _load_team_info(db, team_a_id)
    team_b_info = await _load_team_info(db, team_b_id)

    drivers_a = await _load_team_drivers(db, session_id, team_a_id)
    drivers_b = await _load_team_drivers(db, session_id, team_b_id)

    if not drivers_a:
        raise NotFoundError(
            "session_team",
            f"team={team_a_id} has no drivers in session={session_id}",
        )
    if not drivers_b:
        raise NotFoundError(
            "session_team",
            f"team={team_b_id} has no drivers in session={session_id}",
        )

    # ── Telemetry fetch (cache-first) ────────────────────────────────────────
    client = OpenF1Client(mode="historical")
    laps_df: pd.DataFrame | None = None

    async def _get_data_for_team(
        drivers: list[tuple[int, int, str]],
        team_name: str,
    ) -> list[dict]:
        nonlocal laps_df
        results: list[dict] = []
        for driver_id, car_number, _code in drivers:
            data = await _fetch_driver_data(
                db, session_id, driver_id, car_number,
                client, session_key, laps_df,
            )
            if data is None and laps_df is None:
                # Cache miss — fetch laps once from OpenF1 and retry
                laps_df = client.get_laps(session_key)
                if laps_df.empty:
                    raise UpstreamError(
                        f"OpenF1 returned no laps for session_key={session_key}"
                    )
                data = await _fetch_driver_data(
                    db, session_id, driver_id, car_number,
                    client, session_key, laps_df,
                )
            if data is not None:
                results.append(data)
            else:
                logger.warning(
                    "No telemetry for driver_id=%d (team=%s session=%d) — skipping",
                    driver_id, team_name, session_id,
                )
        return results

    data_a = await _get_data_for_team(drivers_a, team_a_info["name"])
    data_b = await _get_data_for_team(drivers_b, team_b_info["name"])

    if not data_a:
        raise UpstreamError(
            f"No usable telemetry for team '{team_a_info['name']}' in session {session_id}. "
            "Pre-fetch telemetry on the Pipeline page."
        )
    if not data_b:
        raise UpstreamError(
            f"No usable telemetry for team '{team_b_info['name']}' in session {session_id}. "
            "Pre-fetch telemetry on the Pipeline page."
        )

    # ── Corner detection + classification ────────────────────────────────────
    ref_speed = data_a[0]["speed"]

    if circuit.corners and circuit.length_km:
        # Use authoritative FastF1 corner positions — accurate turn count and
        # S/F-aligned distance fractions.  Run seed-circuit-corners to populate.
        circuit_length_m = float(circuit.length_km) * 1000
        raw_corners = cc.corners_from_preloaded(circuit.corners, circuit_length_m, ref_speed)
        logger.info(
            "corner_service: using %d preloaded corners for circuit=%r",
            len(raw_corners), circuit.name,
        )
    else:
        # Fallback: detect corners from local speed minima in the telemetry.
        # Less accurate (may under-count chicanes) but works without seeded data.
        raw_corners = cc.detect_corners_from_speed(ref_speed)
        logger.info(
            "corner_service: no preloaded corners for circuit=%r — using speed-based detection (%d found)",
            circuit.name, len(raw_corners),
        )

    if not raw_corners:
        raise UpstreamError(
            f"Could not detect corners for circuit {circuit.name!r} in session {session_id}. "
            "Run `python -m app.etl seed-circuit-corners` to populate authoritative corner data."
        )

    corners = cc.classify_corners(raw_corners, ref_speed)

    # ── Per-driver metrics → team aggregation ────────────────────────────────
    metrics_a_list = [
        cc.compute_corner_metrics(d["speed"], d["throttle"], d["brake"], corners)
        for d in data_a
    ]
    metrics_b_list = [
        cc.compute_corner_metrics(d["speed"], d["throttle"], d["brake"], corners)
        for d in data_b
    ]

    team_a_metrics = cc.aggregate_team_metrics(metrics_a_list)
    team_b_metrics = cc.aggregate_team_metrics(metrics_b_list)

    # ── Straight detection + per-team speed traces ───────────────────────────
    team_a_speed = cc.aggregate_team_speed([d["speed"] for d in data_a])
    team_b_speed = cc.aggregate_team_speed([d["speed"] for d in data_b])
    straights = cc.detect_straights(corners, circuit.x, circuit.y)

    # ── Summary + figures ────────────────────────────────────────────────────
    summary = cc.build_class_summary(corners, team_a_metrics, team_b_metrics)

    color_a = team_a_info["color_hex"]
    color_b = team_b_info["color_hex"]
    name_a  = team_a_info["name"]
    name_b  = team_b_info["name"]

    v_min_fig = cc.build_v_min_figure(
        corners, team_a_metrics, name_a, color_a,
        team_b_metrics, name_b, color_b,
    )
    class_fig = cc.build_class_summary_figure(
        summary, name_a, color_a, name_b, color_b,
    )
    map_fig = cc.build_corner_track_map_figure(
        circuit.x, circuit.y,
        corners,
        team_a_metrics, name_a, color_a,
        team_b_metrics, name_b, color_b,
    )
    straight_fig = cc.build_straight_performance_figure(
        circuit.x, circuit.y,
        straights,
        team_a_speed, name_a, color_a,
        team_b_speed, name_b, color_b,
    )
    hybrid_fig = cc.build_hybrid_map_figure(
        circuit.x, circuit.y,
        corners,
        team_a_metrics, name_a, color_a,
        team_b_metrics, name_b, color_b,
        straights, team_a_speed, team_b_speed,
    )

    # ── Assemble payload ─────────────────────────────────────────────────────
    return {
        "session_id":            session_id,
        "team_a":                team_a_info,
        "team_b":                team_b_info,
        "corners": [
            {
                "corner_number": c["corner_number"],
                "corner_class":  c["corner_class"],
                "apex_fraction": c["apex_frac"],
                "ref_speed_kmh": c["ref_speed_kmh"],
                "team_a":        a,
                "team_b":        b,
            }
            for c, a, b in zip(corners, team_a_metrics, team_b_metrics)
        ],
        "summary":               summary,
        "v_min_figure":          v_min_fig.to_json(),
        "class_summary_figure":  class_fig.to_json(),
        "track_map_figure":      map_fig.to_json(),
        "straight_map_figure":   straight_fig.to_json(),
        "hybrid_map_figure":     hybrid_fig.to_json(),
    }
