"""Compare two drivers' fastest-lap telemetry traces.

The DB only stores per-lap aggregates (lap_time_ms etc.) — high-frequency
telemetry (speed/throttle/brake per microsector) is NOT persisted. So the
compare endpoint always fetches from the OpenF1 ``/car_data`` endpoint live
via :func:`src.viz._fetch_fastest_lap_openf1`, then builds a simple Plotly
line figure on top of the two returned traces.

This is the same telemetry path the MCP server's ``get_channel_comparison``
tool uses — same OpenF1 calls, same N_POINTS resampling — so the trace
arrays are byte-identical to what the MCP returns. The plotly figure here
is a fresh build (the MCP wraps the Streamlit colour scheme), so JSON
isn't expected to match byte-for-byte, but the data arrays must.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src.openf1 import OpenF1Client

from app.core.errors import BadRequestError, NotFoundError, UpstreamError
from app.db.models import Driver, SessionDriver
from app.db.models import Session as SessionRow

# Number of evenly-spaced distance samples per resampled trace.
# Matches src/features.N_POINTS so the output is byte-comparable with the
# MCP server's get_channel_comparison tool. Inlined here (instead of
# importing from src.features) to avoid pulling Streamlit into the backend.
N_POINTS = 200

Channel = Literal["Speed", "Throttle", "Brake"]


# ---------------------------------------------------------------------------
# Fastest-lap telemetry fetcher (port of src/viz._fetch_fastest_lap_openf1)
# ---------------------------------------------------------------------------


def _fetch_fastest_lap_trace(
    *,
    client: OpenF1Client,
    session_key: int,
    driver_number: int,
    laps_df: pd.DataFrame,
    channel: Channel,
) -> tuple[np.ndarray | None, float | None, int | None]:
    """Same algorithm as ``src/viz._fetch_fastest_lap_openf1``:
    1. Find the driver's fastest non-pit-out lap from ``laps_df``.
    2. Pull ``/car_data`` for the corresponding time window.
    3. Compute a cumulative-distance proxy (speed × dt) and resample the
       requested channel to N_POINTS evenly spaced distance points.

    Returns ``(trace, lap_time_sec, lap_number)`` or ``(None, None, None)``
    on any failure (no laps, no telemetry, single-sample window, etc.).
    """
    col_map = {"Speed": "speed", "Throttle": "throttle", "Brake": "brake"}
    col = col_map.get(channel)
    if col is None:
        return None, None, None

    drv = laps_df[laps_df["driver_number"] == driver_number].copy()
    if "is_pit_out_lap" in drv.columns:
        drv = drv[drv["is_pit_out_lap"] != True]  # noqa: E712
    drv = drv.dropna(subset=["lap_duration"])
    if drv.empty:
        return None, None, None

    fastest = drv.sort_values("lap_duration").iloc[0]
    lap_time = float(fastest["lap_duration"])
    lap_n = int(fastest["lap_number"]) if pd.notna(fastest.get("lap_number")) else None

    date_start = fastest.get("date_start")
    if date_start is None or pd.isna(date_start):
        return None, None, None
    ts_start = pd.Timestamp(date_start)
    ts_end = ts_start + pd.Timedelta(seconds=lap_time)
    date_gte = ts_start.isoformat()
    date_lte = (ts_end + pd.Timedelta(seconds=0.5)).isoformat()

    car = client.get_car_data(
        session_key=session_key,
        driver_number=driver_number,
        date_gte=date_gte,
        date_lte=date_lte,
    )
    if car.empty or col not in car.columns or "date" not in car.columns:
        return None, None, None

    car = car.sort_values("date").reset_index(drop=True)
    values = car[col].to_numpy(dtype=float)
    if len(values) < 2:
        return None, None, None

    dt = car["date"].diff().dt.total_seconds().fillna(0.0).to_numpy(dtype=float)[1:]
    speeds_ms = car["speed"].to_numpy(dtype=float)[:-1] / 3.6
    dist_increments = np.where(np.isfinite(speeds_ms) & np.isfinite(dt), speeds_ms * dt, 0.0)
    dist = np.concatenate([[0.0], np.cumsum(dist_increments)])

    dist_grid = np.linspace(dist[0], dist[-1], N_POINTS)
    trace = np.interp(dist_grid, dist, values)
    return trace, lap_time, lap_n


async def _resolve_session_key(db: AsyncSession, session_id: int) -> int:
    s = await db.get(SessionRow, session_id)
    if s is None:
        raise NotFoundError("session", session_id)
    if s.openf1_session_key is None:
        raise UpstreamError(
            f"session {session_id} has no openf1_session_key — was it loaded by ETL?"
        )
    return int(s.openf1_session_key)


async def _resolve_driver(
    db: AsyncSession, session_id: int, driver_id: int
) -> tuple[int, str]:
    """Return (car_number, code) for a driver in a given session."""
    row = (
        await db.execute(
            select(SessionDriver.car_number, Driver.code)
            .join(Driver, Driver.id == SessionDriver.driver_id)
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
    return int(row.car_number), str(row.code)


def _build_compare_figure(
    *,
    channel: Channel,
    code_a: str,
    code_b: str,
    trace_a: list[float],
    trace_b: list[float],
) -> str:
    """Minimal Plotly line chart of the two channel traces vs distance index.

    Returns the JSON string ready to be parsed by Plotly.js on the client.
    """
    x = list(range(len(trace_a)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=trace_a, mode="lines", name=code_a))
    fig.add_trace(go.Scatter(x=x, y=trace_b, mode="lines", name=code_b))
    fig.update_layout(
        title=f"{channel}: {code_a} vs {code_b} (fastest lap)",
        xaxis_title="Distance index (0..N_POINTS)",
        yaxis_title=channel,
        template="plotly_dark",
        margin={"l": 60, "r": 30, "t": 60, "b": 50},
    )
    return fig.to_json()


async def build_compare_payload(
    db: AsyncSession,
    *,
    session_id: int,
    driver_a_id: int,
    driver_b_id: int,
    channel: Channel,
) -> dict:
    """Fetch fastest-lap traces for two drivers, build a Plotly figure,
    and return the full payload dict ready for serialisation.

    Raises NotFoundError / UpstreamError on missing data.
    """
    if driver_a_id == driver_b_id:
        raise BadRequestError("driver_a and driver_b must differ")
    if channel not in ("Speed", "Throttle", "Brake"):
        raise BadRequestError(f"unsupported channel: {channel}")

    session_key = await _resolve_session_key(db, session_id)
    car_a, code_a = await _resolve_driver(db, session_id, driver_a_id)
    car_b, code_b = await _resolve_driver(db, session_id, driver_b_id)

    # Pull the session's full laps_df from OpenF1 — required by
    # _fetch_fastest_lap_openf1 to identify the fastest lap timestamp window.
    client = OpenF1Client(mode="historical")
    laps_df = client.get_laps(session_key)
    if laps_df.empty:
        raise UpstreamError(
            f"OpenF1 returned no laps for session_key={session_key}"
        )

    def _one(car: int) -> tuple[list[float], float, int | None]:
        trace, lap_time, lap_n = _fetch_fastest_lap_trace(
            client=client,
            session_key=session_key,
            driver_number=car,
            laps_df=laps_df,
            channel=channel,
        )
        if trace is None or lap_time is None:
            raise UpstreamError(
                f"OpenF1 returned no telemetry for driver_number={car} "
                f"channel={channel} session_key={session_key}"
            )
        return [float(v) for v in trace], float(lap_time), lap_n

    trace_a, time_a, lap_a = _one(car_a)
    trace_b, time_b, lap_b = _one(car_b)

    figure_json = _build_compare_figure(
        channel=channel,
        code_a=code_a, code_b=code_b,
        trace_a=trace_a, trace_b=trace_b,
    )

    return {
        "session_id": session_id,
        "channel": channel,
        "driver_a": {
            "driver_id": driver_a_id,
            "car_number": car_a,
            "code": code_a,
            "fastest_lap_time_sec": time_a,
            "fastest_lap_number": lap_a,
            "trace": trace_a,
        },
        "driver_b": {
            "driver_id": driver_b_id,
            "car_number": car_b,
            "code": code_b,
            "fastest_lap_time_sec": time_b,
            "fastest_lap_number": lap_b,
            "trace": trace_b,
        },
        "figure_json": figure_json,
    }
