"""Fetch and cache all car telemetry for a session from OpenF1.

Downloads raw car_data samples for every driver × lap in a session and
stores them in ``car_telemetry``.  Subsequent compare requests read from
the DB instead of calling OpenF1 — typically 100-1000× faster per request.

CLI usage:

    python -m app.etl fetch-telemetry --session-id 73

Strategy:
- One ``get_laps()`` call to get per-driver lap windows.
- One ``get_car_data(session_key, driver_number)`` call per driver
  (no time filter) to get ALL session car_data in a single request.
- Split the full-session blob into per-lap slices in-memory.
- Upsert each slice into ``car_telemetry``.

Total OpenF1 calls: 1 (laps) + N_drivers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, delete, func, select, text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import CarTelemetry
from app.db.models import Session as SessionRow
from app.db.models import SessionDriver, Driver
from app.etl.upserts import upsert_many
from src.openf1 import OpenF1Client

logger = logging.getLogger(__name__)

# Surfaced when OpenF1 no longer recognizes a stored session_key. OpenF1
# renumbered its sessions, so keys written under the old scheme 404 on every
# endpoint. The fix is to re-resolve and update the key in place — NOT to
# re-hydrate, because _upsert_session conflicts on openf1_session_key and a new
# key would insert a duplicate session row rather than repair this one.
_STALE_KEY_MSG = (
    "openf1_session_key={key} is not recognized by OpenF1 (stale key — likely "
    "OpenF1's old numbering). Re-resolve the current key via "
    "get_sessions(year, gp) and UPDATE sessions.openf1_session_key WHERE "
    "id={sid}; do NOT plain re-hydrate (the upsert conflicts on "
    "openf1_session_key and would insert a duplicate session)."
)


@dataclass
class FetchTelemetryResult:
    session_id: int
    drivers_processed: int = 0
    laps_stored: int = 0
    deleted_existing: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "drivers_processed": self.drivers_processed,
            "laps_stored": self.laps_stored,
            "deleted_existing": self.deleted_existing,
            "errors": self.errors,
        }


def _samples_dict(car_df: pd.DataFrame) -> dict:
    """Convert a car_data DataFrame to the columnar JSONB format."""
    return {
        "dates": [d.isoformat() for d in car_df["date"]],
        "speed": car_df["speed"].tolist(),
        "throttle": car_df["throttle"].tolist(),
        "brake": car_df["brake"].tolist(),
        "rpm": car_df["rpm"].tolist(),
        "n_gear": car_df["n_gear"].tolist(),
        "drs": car_df["drs"].tolist(),
    }


class TelemetryFetcher:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.client = OpenF1Client(mode="historical")

    def _load_session(self, session_id: int) -> tuple[int, dict[int, int]]:
        """Return (openf1_session_key, {car_number: driver_id})."""
        row = self.db.get(SessionRow, session_id)
        if row is None:
            raise ValueError(f"session {session_id} not found in DB")
        if row.openf1_session_key is None:
            raise ValueError(
                f"session {session_id} has no openf1_session_key — was it hydrated?"
            )
        session_key = int(row.openf1_session_key)

        sd_rows = self.db.execute(
            select(SessionDriver.car_number, SessionDriver.driver_id)
            .where(SessionDriver.session_id == session_id)
        ).all()
        car_to_driver = {
            int(r.car_number): int(r.driver_id)
            for r in sd_rows
            if r.car_number is not None
        }
        return session_key, car_to_driver

    def _process_driver(
        self,
        session_id: int,
        driver_id: int,
        driver_number: int,
        session_key: int,
        laps_df: pd.DataFrame,
    ) -> int:
        """Fetch all car_data for one driver, split into laps, upsert. Returns laps stored."""
        driver_laps = laps_df[laps_df["driver_number"] == driver_number].copy()
        if "is_pit_out_lap" in driver_laps.columns:
            driver_laps = driver_laps[driver_laps["is_pit_out_lap"] != True]  # noqa: E712
        driver_laps = driver_laps.dropna(subset=["lap_duration", "date_start"])
        if driver_laps.empty:
            logger.warning("driver_number=%d has no usable laps — skipping", driver_number)
            return 0

        # Fetch entire session's car_data for this driver in one API call.
        car_df = self.client.get_car_data(
            session_key=session_key,
            driver_number=driver_number,
        )
        if car_df.empty:
            logger.warning(
                "OpenF1 returned no car_data for driver_number=%d session_key=%d",
                driver_number,
                session_key,
            )
            return 0

        car_df = car_df.sort_values("date").reset_index(drop=True)

        rows: list[dict] = []
        for _, lap in driver_laps.iterrows():
            lap_n = lap.get("lap_number")
            if lap_n is None or pd.isna(lap_n):
                continue
            lap_n = int(lap_n)
            lap_duration = float(lap["lap_duration"])
            ts_start = pd.Timestamp(lap["date_start"])
            ts_end = ts_start + pd.Timedelta(seconds=lap_duration + 0.5)

            # Slice car_data for this lap window.
            mask = (car_df["date"] >= ts_start) & (car_df["date"] <= ts_end)
            lap_car = car_df[mask]
            if len(lap_car) < 2:
                continue

            rows.append({
                "session_id": session_id,
                "driver_id": driver_id,
                "lap_number": lap_n,
                "lap_duration": lap_duration,
                "samples": _samples_dict(lap_car),
            })

        if not rows:
            return 0

        upsert_many(
            self.db,
            CarTelemetry.__table__,
            rows=rows,
            conflict_cols=["session_id", "driver_id", "lap_number"],
            update_cols=["lap_duration", "samples", "fetched_at"],
        )
        return len(rows)

    def _purge_existing(self, session_id: int) -> int:
        """Delete any previously-cached telemetry for this session.

        Runs inside the same transaction as the re-fetch (the only commit is at
        the end of :meth:`fetch_session`), so a fetch that bails early — e.g. a
        stale-key 404 — rolls the delete back and leaves the old cache intact.
        Only once we have real laps to fetch do we wipe and start fresh, which
        guarantees stale or partial rows from a previous run never linger.
        """
        existing = self.db.execute(
            select(func.count())
            .select_from(CarTelemetry)
            .where(CarTelemetry.session_id == session_id)
        ).scalar_one()
        if existing:
            self.db.execute(
                delete(CarTelemetry).where(CarTelemetry.session_id == session_id)
            )
            logger.info(
                "session %d: purged %d existing telemetry rows before re-fetch",
                session_id, existing,
            )
        return int(existing)

    def fetch_session(self, session_id: int) -> FetchTelemetryResult:
        result = FetchTelemetryResult(session_id=session_id)

        try:
            session_key, car_to_driver = self._load_session(session_id)
        except ValueError as exc:
            result.errors.append(str(exc))
            return result

        print(f"session_key={session_key}  drivers={len(car_to_driver)}", flush=True)

        laps_df = self.client.get_laps(session_key)
        if laps_df.empty:
            # Empty laps could mean a genuinely lap-less session OR a stale
            # session_key that 404s. Disambiguate with a cheap existence check
            # (only on this error path, so the happy path keeps its 1 + N call
            # budget).
            if not self.client.session_exists(session_key):
                logger.warning(
                    "session %d: openf1_session_key=%d not found on OpenF1 (stale key)",
                    session_id, session_key,
                )
                result.errors.append(
                    _STALE_KEY_MSG.format(sid=session_id, key=session_key)
                )
            else:
                result.errors.append(
                    f"OpenF1 returned no laps for session_key={session_key}"
                )
            return result

        # Telemetry exists for this session? Wipe it and re-fetch from scratch so
        # stale/partial rows from a previous run can't survive. Safe to do here:
        # we have confirmed real laps, and the whole job commits atomically below.
        result.deleted_existing = self._purge_existing(session_id)
        if result.deleted_existing:
            print(
                f"  purged {result.deleted_existing} existing telemetry rows — "
                "fetching fresh",
                flush=True,
            )

        for driver_number, driver_id in sorted(car_to_driver.items()):
            print(f"  driver_number={driver_number} driver_id={driver_id} …", flush=True)
            try:
                stored = self._process_driver(
                    session_id=session_id,
                    driver_id=driver_id,
                    driver_number=driver_number,
                    session_key=session_key,
                    laps_df=laps_df,
                )
                result.laps_stored += stored
                result.drivers_processed += 1
                print(f"    stored {stored} laps", flush=True)
            except Exception as exc:
                msg = f"driver_number={driver_number}: {exc}"
                logger.exception("etl.fetch_telemetry error: %s", msg)
                result.errors.append(msg)

        # Mark session as fully fetched.
        self.db.execute(
            text("UPDATE sessions SET telemetry_fetched_at = NOW() WHERE id = :sid"),
            {"sid": session_id},
        )
        self.db.commit()
        return result


def run(*, session_id: int) -> FetchTelemetryResult:
    engine = create_engine(settings.DATABASE_URL_SYNC, future=True)
    with Session(engine) as db:
        try:
            fetcher = TelemetryFetcher(db)
            return fetcher.fetch_session(session_id)
        except Exception:
            db.rollback()
            raise
