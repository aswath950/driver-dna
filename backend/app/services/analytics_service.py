"""DB → ``RaceAnalyser`` adapter.

Phase 7 endpoints reuse the existing race-analytics code in
``src/race_engine.py`` instead of re-implementing it. The only new code
here is a thin adapter that pulls ``lap_times`` / ``session_drivers`` /
``race_results`` from Postgres and shapes them into the three DataFrames
``RaceAnalyser`` expects:

    laps     : driver_number, lap_number, lap_duration, is_pit_out_lap
    stints   : driver_number, stint_number, compound, lap_start, lap_end
    position : driver_number, position, date

Notes
-----
- ``driver_number`` in race_engine = our ``session_drivers.car_number``.
- Our DB has no telemetry traces (only lap_times) — compare endpoints
  fetch live from OpenF1 via :mod:`app.services.compare_service`.
- The DB only has each driver's FINAL classification in ``race_results``,
  not their per-lap position. We synthesise a per-lap position table by
  cumulative race time, derived from lap_durations — sufficient for
  ``RaceAnalyser`` undercut/overcut detection.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src.race_engine import RaceAnalyser

from app.core.errors import UpstreamError
from app.db.models import LapTime, SessionDriver
from app.db.models import Session as SessionRow


async def _load_car_number_map(
    db: AsyncSession, session_id: int
) -> dict[int, int]:
    """Return {driver_id: car_number} for the given session."""
    rows = (
        await db.execute(
            select(SessionDriver.driver_id, SessionDriver.car_number).where(
                SessionDriver.session_id == session_id
            )
        )
    ).all()
    return {int(r.driver_id): int(r.car_number) for r in rows}


async def _load_laps_df(
    db: AsyncSession, session_id: int, drv_map: dict[int, int]
) -> pd.DataFrame:
    rows = (
        await db.execute(
            select(
                LapTime.driver_id,
                LapTime.lap_number,
                LapTime.lap_time_ms,
                LapTime.is_pit_out,
                LapTime.is_pit_in,
                LapTime.compound,
                LapTime.tyre_life,
            ).where(LapTime.session_id == session_id)
        )
    ).all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=[
        "driver_id", "lap_number", "lap_time_ms",
        "is_pit_out", "is_pit_in", "compound", "tyre_life",
    ])
    df["driver_number"] = df["driver_id"].map(drv_map)
    df = df.dropna(subset=["driver_number"]).copy()
    df["driver_number"] = df["driver_number"].astype(int)
    df["lap_duration"] = df["lap_time_ms"] / 1000.0
    df["is_pit_out_lap"] = df["is_pit_out"].astype(bool)
    # Unwrap CompoundType enum so downstream code (and JSON serialisation)
    # gets a plain string like "SOFT" rather than "CompoundType.SOFT".
    df["compound"] = df["compound"].map(lambda v: v.value if hasattr(v, "value") else str(v))
    return df[[
        "driver_number", "lap_number", "lap_duration",
        "is_pit_out_lap", "compound", "tyre_life",
    ]].sort_values(["driver_number", "lap_number"]).reset_index(drop=True)


def _build_stints_df(laps_df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct stints by collapsing consecutive same-compound runs per
    driver. Mirrors the structure OpenF1's /stints endpoint returns.
    """
    if laps_df.empty:
        return pd.DataFrame(
            columns=["driver_number", "stint_number", "compound",
                     "tyre_age_at_start", "lap_start", "lap_end"]
        )
    out: list[dict] = []
    for drv, grp in laps_df.groupby("driver_number"):
        grp = grp.sort_values("lap_number").reset_index(drop=True)
        compounds = grp["compound"].astype(str).tolist()
        laps = grp["lap_number"].tolist()
        ages = grp["tyre_life"].fillna(0).astype(int).tolist()
        if not compounds:
            continue
        stint_n = 1
        seg_compound = compounds[0]
        seg_start = laps[0]
        seg_age_start = ages[0]
        for i in range(1, len(compounds)):
            if compounds[i] != seg_compound:
                out.append({
                    "driver_number": int(drv),
                    "stint_number": stint_n,
                    "compound": seg_compound,
                    "tyre_age_at_start": int(seg_age_start),
                    "lap_start": int(seg_start),
                    "lap_end": int(laps[i - 1]),
                })
                stint_n += 1
                seg_compound = compounds[i]
                seg_start = laps[i]
                seg_age_start = ages[i]
        out.append({
            "driver_number": int(drv),
            "stint_number": stint_n,
            "compound": seg_compound,
            "tyre_age_at_start": int(seg_age_start),
            "lap_start": int(seg_start),
            "lap_end": int(laps[-1]),
        })
    return pd.DataFrame(out)


async def _load_session_start(
    db: AsyncSession, session_id: int
) -> datetime:
    s = await db.get(SessionRow, session_id)
    if s and s.date_start:
        return s.date_start
    return datetime(2024, 1, 1, tzinfo=UTC)


def _synthesise_position_df(
    laps_df: pd.DataFrame, session_start: datetime
) -> pd.DataFrame:
    """Build a per-lap position table from cumulative race time.

    We don't store OpenF1 ``/position`` rows in the DB, so reconstruct an
    equivalent: for each lap N, the driver with the smallest cumulative
    race time after N laps is in P1, the next is P2, etc.

    Returns
    -------
    DataFrame with one row per (driver_number, lap_number):
        driver_number, position, date
    """
    if laps_df.empty:
        return pd.DataFrame(columns=["driver_number", "position", "date"])

    df = laps_df.dropna(subset=["lap_duration"]).copy()
    df = df[df["lap_duration"] > 0]
    df = df.sort_values(["driver_number", "lap_number"])
    df["cum_time"] = df.groupby("driver_number")["lap_duration"].cumsum()

    rows: list[dict] = []
    for lap_n, lap_grp in df.groupby("lap_number"):
        ranked = lap_grp.sort_values("cum_time").reset_index(drop=True)
        for pos, r in enumerate(ranked.itertuples(index=False), start=1):
            rows.append({
                "driver_number": int(r.driver_number),
                "position": int(pos),
                "date": session_start + timedelta(seconds=float(r.cum_time)),
            })
    return pd.DataFrame(rows)


async def build_analyser(db: AsyncSession, session_id: int) -> RaceAnalyser:
    """Construct a :class:`RaceAnalyser` from DB rows for one session.

    Raises :class:`UpstreamError` (503) if the session has not been hydrated.
    """
    drv_map = await _load_car_number_map(db, session_id)
    laps_df = await _load_laps_df(db, session_id, drv_map)
    if laps_df.empty:
        raise UpstreamError(
            f"session {session_id} has no lap data hydrated yet — "
            "run `python -m app.etl hydrate ...`"
        )
    stints_df = _build_stints_df(laps_df)
    start = await _load_session_start(db, session_id)
    position_df = _synthesise_position_df(laps_df, start)
    return RaceAnalyser(laps_df, stints_df, position_df)


# ---------------------------------------------------------------------------
# Result projection helpers — convert RaceAnalyser DataFrame outputs into
# tabular records the router serialises directly.
# ---------------------------------------------------------------------------


def rolling_pace_rows(
    analyser: RaceAnalyser, *, window: int, drv_map: dict[int, int]
) -> list[dict]:
    df = analyser.rolling_pace(window=window)
    if df.empty:
        return []
    num_to_id = {v: k for k, v in drv_map.items()}
    out: list[dict] = []
    for lap, row in df.iterrows():
        for drv_num, val in row.items():
            if pd.isna(val):
                continue
            did = num_to_id.get(int(drv_num))
            if did is None:
                continue
            out.append({
                "driver_id": did,
                "lap": int(lap),
                "rolling_sec": float(val),
            })
    return out


def gap_to_leader_rows(
    analyser: RaceAnalyser, *, drv_map: dict[int, int]
) -> list[dict]:
    df = analyser.gap_to_leader()
    if df.empty:
        return []
    num_to_id = {v: k for k, v in drv_map.items()}
    out: list[dict] = []
    for lap, row in df.iterrows():
        for drv_num, val in row.items():
            if pd.isna(val):
                continue
            did = num_to_id.get(int(drv_num))
            if did is None:
                continue
            out.append({
                "driver_id": did,
                "lap": int(lap),
                "gap_sec": float(val),
            })
    return out


def undercut_events(
    analyser: RaceAnalyser, *, drv_map: dict[int, int]
) -> list[dict]:
    events = analyser.detect_undercuts()
    num_to_id = {v: k for k, v in drv_map.items()}
    out: list[dict] = []
    for e in events:
        atk = num_to_id.get(int(e.get("attacking_driver", -1)))
        defn = num_to_id.get(int(e.get("defending_driver", -1)))
        if atk is None or defn is None:
            continue
        out.append({
            "lap": int(e["lap"]),
            "attacker_id": atk,
            "victim_id": defn,
            "type": str(e.get("type", "undercut")),
        })
    return out


def degradation_rows(
    analyser: RaceAnalyser, *, drv_map: dict[int, int]
) -> list[dict]:
    df = analyser.tyre_degradation()
    if df.empty:
        return []
    num_to_id = {v: k for k, v in drv_map.items()}
    out: list[dict] = []
    for r in df.itertuples(index=False):
        did = num_to_id.get(int(r.driver_number))
        if did is None:
            continue
        out.append({
            "driver_id": did,
            "stint": int(r.stint_number),
            "compound": str(r.compound),
            "laps_in_stint": int(r.laps_in_stint),
            "deg_sec_per_lap": float(r.deg_per_lap)
                if pd.notna(r.deg_per_lap) else 0.0,
            "mean_pace_sec": float(r.mean_pace)
                if pd.notna(r.mean_pace) else 0.0,
        })
    return out


__all__ = [
    "build_analyser",
    "_load_car_number_map",
    "rolling_pace_rows",
    "gap_to_leader_rows",
    "undercut_events",
    "degradation_rows",
]
