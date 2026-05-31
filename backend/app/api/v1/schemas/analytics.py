from __future__ import annotations

from typing import Literal

from app.api.v1.schemas.common import ORMModel


class RollingPaceRow(ORMModel):
    driver_id: int
    lap: int
    rolling_sec: float


class GapRow(ORMModel):
    driver_id: int
    lap: int
    gap_sec: float


class UndercutEvent(ORMModel):
    lap: int
    attacker_id: int
    victim_id: int
    type: Literal["undercut", "overcut"]


class DegradationRow(ORMModel):
    driver_id: int
    stint: int
    compound: str
    laps_in_stint: int
    deg_sec_per_lap: float
    mean_pace_sec: float


# --- Compare ---------------------------------------------------------------


Channel = Literal["Speed", "Throttle", "Brake"]


class CompareDriverTrace(ORMModel):
    driver_id: int
    car_number: int
    code: str
    fastest_lap_time_sec: float
    fastest_lap_number: int | None = None
    # N_POINTS evenly-spaced distance-resampled trace.
    trace: list[float]


class ComparePayload(ORMModel):
    session_id: int
    channel: Channel
    driver_a: CompareDriverTrace
    driver_b: CompareDriverTrace
    # Plotly figure JSON (string) — frontend renders directly with Plotly.js.
    figure_json: str
