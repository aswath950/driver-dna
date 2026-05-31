"""Hydrate a single race-weekend session (or all sessions for a weekend)
into Postgres from the OpenF1 API. Idempotent — safe to re-run.

CLI usage:

    python -m app.etl hydrate --year 2024 --gp "Monaco" --session R
    python -m app.etl hydrate --year 2024 --gp "Monaco" --all-sessions
    python -m app.etl hydrate --year 2024 --gp "Monaco" --session R --dry-run

Returns a summary dict with row counts per table; the CLI prints it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from src.openf1 import OpenF1Client

from app.core.config import settings
from app.db.models import (
    CompoundType,
    Driver,
    Event,
    LapTime,
    RaceResult,
    Season,
    SessionDriver,
    SessionType,
    Team,
)
from app.db.models import Session as SessionRow
from app.etl.upserts import upsert_many, upsert_one

logger = logging.getLogger(__name__)


# F1 points table (top 10). Index 0 = 1st place.
POINTS_TABLE: list[float] = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]


def _points_for(position: int | None) -> float:
    if position is None or position < 1 or position > len(POINTS_TABLE):
        return 0.0
    return POINTS_TABLE[position - 1]


def _to_session_type(raw: str | None) -> SessionType | None:
    """Map OpenF1 session_type/session_name to our enum."""
    if not raw:
        return None
    s = str(raw).strip().lower()
    mapping = {
        "practice 1": SessionType.FP1, "fp1": SessionType.FP1,
        "practice 2": SessionType.FP2, "fp2": SessionType.FP2,
        "practice 3": SessionType.FP3, "fp3": SessionType.FP3,
        "qualifying": SessionType.Q,   "q": SessionType.Q,
        "sprint qualifying": SessionType.SQ, "sq": SessionType.SQ,
        "sprint shootout": SessionType.SQ,
        "sprint": SessionType.S,        "s": SessionType.S,
        "race": SessionType.R,          "r": SessionType.R,
    }
    return mapping.get(s)


def _to_compound(raw: Any) -> CompoundType:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return CompoundType.UNKNOWN
    try:
        return CompoundType(str(raw).strip().upper())
    except ValueError:
        return CompoundType.UNKNOWN


@dataclass
class HydrationResult:
    season_id: int | None = None
    event_id: int | None = None
    session_ids: list[int] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    dry_run: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "season_id": self.season_id,
            "event_id": self.event_id,
            "session_ids": self.session_ids,
            "counts": self.counts,
            "dry_run": self.dry_run,
        }


class SessionHydrator:
    """Hydrates one race weekend's sessions into Postgres.

    Stateless — pass in the OpenF1 client and a DB session.
    """

    def __init__(self, db: Session, client: OpenF1Client | None = None) -> None:
        self.db = db
        self.client = client or OpenF1Client(mode="historical")

    # ------------------------------------------------------------------
    # Reference data
    # ------------------------------------------------------------------

    def _upsert_season(self, year: int) -> int:
        row = upsert_one(
            self.db,
            Season.__table__,
            values={"year": year},
            conflict_cols=["year"],
            update_cols=[],
        )
        return int(row["id"])

    def _upsert_default_circuit(self) -> int:
        """For v1 we don't have circuit geometry per meeting in OpenF1 cheaply;
        fall back to a single 'Unknown' circuit. Phase 6+ can swap this for a
        proper circuits sync."""
        row = upsert_one(
            self.db,
            __import__("app.db.models", fromlist=["Circuit"]).Circuit.__table__,
            values={"name": "Unknown"},
            conflict_cols=["name"],
            update_cols=[],
            returning_cols=("id",),
        )
        return int(row["id"])

    def _upsert_event(
        self,
        *,
        season_id: int,
        circuit_id: int,
        round_n: int,
        name: str,
        start_date,
    ) -> int:
        row = upsert_one(
            self.db,
            Event.__table__,
            values={
                "season_id": season_id,
                "circuit_id": circuit_id,
                "round": round_n,
                "name": name,
                "start_date": start_date,
            },
            conflict_cols=["season_id", "round"],
            update_cols=["circuit_id", "name", "start_date"],
        )
        return int(row["id"])

    def _upsert_session(
        self,
        *,
        event_id: int,
        type_: SessionType,
        date_start,
        openf1_session_key: int,
    ) -> int:
        row = upsert_one(
            self.db,
            SessionRow.__table__,
            values={
                "event_id": event_id,
                "type": type_,
                "date_start": date_start,
                "openf1_session_key": openf1_session_key,
            },
            conflict_cols=["openf1_session_key"],
            update_cols=["event_id", "type", "date_start"],
        )
        return int(row["id"])

    # ------------------------------------------------------------------
    # Drivers + teams
    # ------------------------------------------------------------------

    def _upsert_team(self, name: str | None, color: str | None) -> int | None:
        if not name or (isinstance(name, float) and pd.isna(name)):
            return None
        color_hex = None
        if color and not (isinstance(color, float) and pd.isna(color)):
            color_hex = "#" + str(color).lstrip("#")[:6].upper()
        row = upsert_one(
            self.db,
            Team.__table__,
            values={"name": str(name), "color_hex": color_hex},
            conflict_cols=["name"],
            update_cols=["color_hex"],
        )
        return int(row["id"])

    def _upsert_driver(
        self, code: str, full_name: str, team_id: int | None
    ) -> int:
        row = upsert_one(
            self.db,
            Driver.__table__,
            values={
                "code": code,
                "full_name": full_name,
                "current_team_id": team_id,
            },
            conflict_cols=["code"],
            update_cols=["full_name", "current_team_id"],
        )
        return int(row["id"])

    def _sync_drivers(
        self, session_id: int, drivers_df: pd.DataFrame
    ) -> dict[int, int]:
        """Returns {driver_number → drivers.id}."""
        out: dict[int, int] = {}
        sd_rows: list[dict[str, Any]] = []
        for _, r in drivers_df.iterrows():
            num = r.get("driver_number")
            code = r.get("name_acronym")
            full = r.get("full_name") or code or "Unknown"
            if pd.isna(num) or not code or pd.isna(code):
                continue
            num = int(num)
            team_id = self._upsert_team(r.get("team_name"), r.get("team_colour"))
            if team_id is None:
                # Driver without a team — skip; session_drivers FK requires team.
                logger.warning("driver %s has no team, skipping", code)
                continue
            driver_id = self._upsert_driver(str(code), str(full), team_id)
            out[num] = driver_id
            sd_rows.append({
                "session_id": session_id,
                "driver_id": driver_id,
                "team_id": team_id,
                "car_number": num,
            })
        if sd_rows:
            upsert_many(
                self.db,
                SessionDriver.__table__,
                rows=sd_rows,
                conflict_cols=["session_id", "driver_id"],
                update_cols=["team_id", "car_number"],
            )
        return out

    # ------------------------------------------------------------------
    # Laps + stints
    # ------------------------------------------------------------------

    def _sync_laps(
        self,
        session_id: int,
        laps_df: pd.DataFrame,
        stints_df: pd.DataFrame,
        num_to_driver_id: dict[int, int],
    ) -> int:
        if laps_df.empty:
            return 0

        # Build (driver_number, lap_number) → (compound, tyre_life) lookup
        # from stints, so we can attach compound info during the lap upsert.
        compound_lookup: dict[tuple[int, int], tuple[CompoundType, int | None]] = {}
        if not stints_df.empty:
            for _, st in stints_df.iterrows():
                drv = st.get("driver_number")
                lap_start = st.get("lap_start")
                lap_end = st.get("lap_end")
                if pd.isna(drv) or pd.isna(lap_start) or pd.isna(lap_end):
                    continue
                drv = int(drv)
                comp = _to_compound(st.get("compound"))
                age_at_start = st.get("tyre_age_at_start")
                age_at_start = int(age_at_start) if not pd.isna(age_at_start) else 0
                for lap_n in range(int(lap_start), int(lap_end) + 1):
                    tyre_life = age_at_start + (lap_n - int(lap_start))
                    compound_lookup[(drv, lap_n)] = (comp, tyre_life)

        rows: list[dict[str, Any]] = []
        for _, lap in laps_df.iterrows():
            drv = lap.get("driver_number")
            lap_n = lap.get("lap_number")
            dur = lap.get("lap_duration")
            if pd.isna(drv) or pd.isna(lap_n) or int(lap_n) < 1:
                continue
            drv = int(drv)
            driver_id = num_to_driver_id.get(drv)
            if driver_id is None:
                continue
            lap_time_ms = (
                int(round(float(dur) * 1000)) if not pd.isna(dur) and dur > 0 else None
            )
            compound, tyre_life = compound_lookup.get(
                (drv, int(lap_n)), (CompoundType.UNKNOWN, None)
            )
            is_pit_out = bool(lap.get("is_pit_out_lap")) if not pd.isna(
                lap.get("is_pit_out_lap")
            ) else False
            rows.append({
                "session_id": session_id,
                "driver_id": driver_id,
                "lap_number": int(lap_n),
                "lap_time_ms": lap_time_ms,
                "compound": compound,
                "tyre_life": tyre_life,
                "is_pit_out": is_pit_out,
                "is_pit_in": False,  # OpenF1 doesn't expose this directly
            })
        return upsert_many(
            self.db,
            LapTime.__table__,
            rows=rows,
            conflict_cols=["session_id", "driver_id", "lap_number"],
            update_cols=[
                "lap_time_ms", "compound", "tyre_life", "is_pit_out", "is_pit_in",
            ],
        )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def _sync_results(
        self,
        session_id: int,
        position_df: pd.DataFrame,
        laps_df: pd.DataFrame,
        num_to_driver_id: dict[int, int],
        type_: SessionType,
    ) -> int:
        """Derive final classification from the LAST position row per driver."""
        if position_df.empty:
            return 0

        # Last position per driver (chronologically).
        pos_sorted = position_df.sort_values("date") if "date" in position_df.columns else position_df
        last_per_driver = (
            pos_sorted.dropna(subset=["driver_number", "position"])
            .groupby("driver_number")
            .last()
            .reset_index()
        )

        # Grid = first position seen for that driver in this session.
        first_per_driver = (
            pos_sorted.dropna(subset=["driver_number", "position"])
            .groupby("driver_number")
            .first()
            .reset_index()
            .set_index("driver_number")
        )

        # Fastest lap per driver
        fastest: dict[int, int | None] = {}
        if not laps_df.empty:
            for drv, sub in laps_df.dropna(subset=["lap_duration"]).groupby("driver_number"):
                if pd.isna(drv):
                    continue
                drv_int = int(drv)
                best = sub["lap_duration"].min()
                if pd.notna(best) and best > 0:
                    fastest[drv_int] = int(round(float(best) * 1000))

        rows: list[dict[str, Any]] = []
        for _, r in last_per_driver.iterrows():
            drv_num = int(r["driver_number"])
            driver_id = num_to_driver_id.get(drv_num)
            if driver_id is None:
                continue
            position = int(r["position"]) if not pd.isna(r["position"]) else None
            grid_val = first_per_driver.loc[drv_num, "position"] if drv_num in first_per_driver.index else None
            grid = int(grid_val) if grid_val is not None and not pd.isna(grid_val) else None
            # Only races award points.
            points = _points_for(position) if type_ in (SessionType.R, SessionType.S) else 0.0
            # Sprint = half points for top 8 in modern era; keep simple in v1.
            rows.append({
                "session_id": session_id,
                "driver_id": driver_id,
                "position": position,
                "grid": grid,
                "points": points,
                "status": "Finished",
                "fastest_lap_ms": fastest.get(drv_num),
            })
        return upsert_many(
            self.db,
            RaceResult.__table__,
            rows=rows,
            conflict_cols=["session_id", "driver_id"],
            update_cols=["position", "grid", "points", "status", "fastest_lap_ms"],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def hydrate(
        self,
        *,
        year: int,
        grand_prix: str,
        session_type: str | None = None,
        round_hint: int | None = None,
        dry_run: bool = False,
    ) -> HydrationResult:
        """Hydrate either one session (when ``session_type`` given) or all
        sessions for the weekend (when ``session_type`` is None).
        """
        result = HydrationResult(dry_run=dry_run)
        result.counts = {
            "session_drivers": 0, "lap_times": 0, "race_results": 0,
        }

        sessions_df = self.client.get_sessions(year, grand_prix)
        if sessions_df.empty:
            raise ValueError(f"OpenF1 returned no sessions for {grand_prix} {year}")

        # Filter by session_type if requested.
        targets = sessions_df.copy()
        if session_type:
            wanted = _to_session_type(session_type)
            if wanted is None:
                raise ValueError(f"unknown session_type: {session_type!r}")
            # pandas Series == Enum compares oddly (always False); use .apply().
            mask_name = targets["session_name"].apply(lambda v: _to_session_type(v) == wanted)
            mask = mask_name
            if "session_type" in targets.columns:
                mask_type = targets["session_type"].apply(lambda v: _to_session_type(v) == wanted)
                mask = mask_name | mask_type
            targets = targets[mask]
            if targets.empty:
                raise ValueError(f"no {session_type!r} session found for {grand_prix} {year}")

        # Common reference data (season, circuit, event) — derived from the
        # first matching session.
        first = targets.iloc[0]
        season_id = self._upsert_season(year)
        circuit_id = self._upsert_default_circuit()
        round_n = int(first.get("meeting_key", 0)) if round_hint is None else round_hint
        if round_n == 0 and round_hint is None:
            # meeting_key works as a unique-per-season round substitute when we
            # don't have round info.
            round_n = int(first.get("meeting_key", 1))
        event_id = self._upsert_event(
            season_id=season_id,
            circuit_id=circuit_id,
            round_n=round_n,
            name=str(first.get("meeting_name") or grand_prix),
            start_date=(
                first["date_start"].date()
                if "date_start" in first and pd.notna(first.get("date_start"))
                else None
            ),
        )
        result.season_id = season_id
        result.event_id = event_id

        for _, srow in targets.iterrows():
            sk = int(srow["session_key"])
            ttype = _to_session_type(srow.get("session_name")) or _to_session_type(
                srow.get("session_type")
            )
            if ttype is None:
                logger.warning("skipping session_key=%s — unknown session_name=%r",
                               sk, srow.get("session_name"))
                continue
            sid = self._upsert_session(
                event_id=event_id,
                type_=ttype,
                date_start=srow.get("date_start"),
                openf1_session_key=sk,
            )
            result.session_ids.append(sid)

            drivers_df = self.client.get_drivers(sk)
            num_to_driver = self._sync_drivers(sid, drivers_df)
            result.counts["session_drivers"] += len(num_to_driver)

            laps_df = self.client.get_laps(sk)
            stints_df = self.client.get_stints(sk)
            n_laps = self._sync_laps(sid, laps_df, stints_df, num_to_driver)
            result.counts["lap_times"] += n_laps

            position_df = self.client.get_position(sk)
            n_results = self._sync_results(
                sid, position_df, laps_df, num_to_driver, ttype
            )
            result.counts["race_results"] += n_results

        return result


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def run(
    *,
    year: int,
    grand_prix: str,
    session_type: str | None = None,
    dry_run: bool = False,
) -> HydrationResult:
    """Open a sync DB session, hydrate the requested session(s), then commit
    (or rollback on dry-run). One transaction wraps the whole job — partial
    failures leave the DB untouched.
    """
    engine = create_engine(settings.DATABASE_URL_SYNC, future=True)
    with Session(engine) as db:
        try:
            hydrator = SessionHydrator(db)
            result = hydrator.hydrate(
                year=year,
                grand_prix=grand_prix,
                session_type=session_type,
                dry_run=dry_run,
            )
            if dry_run:
                db.rollback()
                logger.info("etl.hydrate.dry_run", extra={"summary": result.as_dict()})
            else:
                db.commit()
        except Exception:
            db.rollback()
            raise
    return result
