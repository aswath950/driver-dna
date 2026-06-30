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


Channel = Literal[
    "Speed", "Throttle", "Brake", "RPM", "nGear", "DRS",
    "TimeDelta", "SpeedTimeDelta", "TrackMap", "SectorTimes",
]


# --- Sector Times ------------------------------------------------------------


class SectorDriverSplits(ORMModel):
    driver_id: int
    code: str
    lap_number: int | None = None
    lap_time_ms: int | None = None
    sector1_ms: int | None = None
    sector2_ms: int | None = None
    sector3_ms: int | None = None


class SectorTimesPayload(ORMModel):
    session_id: int
    driver_a: SectorDriverSplits
    driver_b: SectorDriverSplits
    figure_json: str


# --- Track Map ---------------------------------------------------------------


class TrackMapDriverTrace(ORMModel):
    driver_id: int
    code: str


class TrackMapPayload(ORMModel):
    session_id: int
    driver_a: TrackMapDriverTrace
    driver_b: TrackMapDriverTrace
    # Circuit outline shared by both drivers (from circuits.x / circuits.y).
    circuit_x: list[float]
    circuit_y: list[float]
    figure_json: str


# --- Compare -----------------------------------------------------------------


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


# --- Corner Performance -------------------------------------------------------


class CornerMetrics(ORMModel):
    v_min: float
    exit_speed: float
    throttle_dist_frac: float
    brake_point_frac: float


class TeamInfo(ORMModel):
    id: int
    name: str
    color_hex: str


class SingleCorner(ORMModel):
    corner_number: int
    corner_class: Literal["slow", "medium", "high"]
    apex_fraction: float
    ref_speed_kmh: float
    team_a: CornerMetrics
    team_b: CornerMetrics


class ClassSummary(ORMModel):
    corner_count: int
    team_a: CornerMetrics
    team_b: CornerMetrics


class CornerPerformancePayload(ORMModel):
    session_id: int
    team_a: TeamInfo
    team_b: TeamInfo
    corners: list[SingleCorner]
    summary: dict[str, ClassSummary]   # keys: "slow", "medium", "high"
    v_min_figure: str                  # Plotly JSON
    class_summary_figure: str          # Plotly JSON
    track_map_figure: str              # Plotly JSON
