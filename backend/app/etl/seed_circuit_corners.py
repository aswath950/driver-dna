"""Seed official corner data for all circuits from FastF1.

For each circuit in the database that has at least one event in the target
year, fetches the official turn list (number, letter, distance from S/F line
in metres) and stores it in ``circuits.corners``.  Also back-fills
``circuits.length_km`` from the marshal-sector data when the column is NULL.

This is idempotent — re-running the command overwrites existing corner data
with the latest FastF1 values.

CLI usage:
    python -m app.etl seed-circuit-corners
    python -m app.etl seed-circuit-corners --year 2024
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import fastf1
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Circuit, Event
from app.db.models import Season
from app.etl.upserts import upsert_one

logger = logging.getLogger(__name__)


@dataclass
class SeedCornersResult:
    year: int
    circuits_updated: int
    circuits_skipped: int
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "year": self.year,
            "circuits_updated": self.circuits_updated,
            "circuits_skipped": self.circuits_skipped,
            "errors": self.errors,
        }


def _fetch_circuit_info(year: int, round_num: int) -> tuple[list[dict], float | None] | None:
    """Return (corners, total_distance_m) for one event via FastF1.

    Uses the minimum load (no telemetry, laps, weather, or messages) so only
    the circuit metadata file is downloaded.  Returns None on any error.
    """
    try:
        session = fastf1.get_session(year, round_num, "R")
        session.load(telemetry=False, laps=False, weather=False, messages=False)
        info = session.get_circuit_info()

        corners = [
            {
                "number": int(row["Number"]),
                "letter": str(row["Letter"]).strip(),
                "distance_m": float(row["Distance"]),
            }
            for _, row in info.corners.iterrows()
        ]

        # Marshal sectors cover the full lap; the last Distance value is the
        # total circuit length in metres.
        total_m: float | None = None
        if not info.marshal_sectors.empty:
            total_m = float(info.marshal_sectors["Distance"].max())

        return corners, total_m

    except Exception as exc:
        logger.warning(
            "FastF1 fetch failed year=%d round=%d: %s", year, round_num, exc
        )
        return None


def run(*, year: int = 2024) -> SeedCornersResult:
    """Fetch and store corner data for all circuits with events in ``year``."""
    # Point FastF1 at a temp cache so the ETL is self-contained.
    cache_dir = tempfile.mkdtemp(prefix="fastf1_")
    fastf1.Cache.enable_cache(cache_dir)

    engine = create_engine(settings.DATABASE_URL_SYNC, future=True)
    result = SeedCornersResult(year=year, circuits_updated=0, circuits_skipped=0)

    with Session(engine) as db:
        # Collect one representative event per circuit for the target year.
        rows = db.execute(
            select(Circuit.id, Circuit.name, Circuit.length_km, Event.round)
            .join(Event, Event.circuit_id == Circuit.id)
            .join(Season, Season.id == Event.season_id)
            .where(Season.year == year)
            .order_by(Circuit.id, Event.round)
        ).all()

        seen: dict[int, tuple[str, Decimal | None, int]] = {}
        for circuit_id, circuit_name, length_km, round_num in rows:
            if circuit_id not in seen:
                seen[circuit_id] = (circuit_name, length_km, int(round_num))

        try:
            for circuit_id, (circuit_name, length_km, round_num) in seen.items():
                logger.info(
                    "Fetching corners  circuit=%r  round=%d  year=%d",
                    circuit_name, round_num, year,
                )
                data = _fetch_circuit_info(year, round_num)
                if data is None:
                    logger.warning("Skipping circuit=%r — FastF1 fetch failed", circuit_name)
                    result.circuits_skipped += 1
                    result.errors.append(f"{circuit_name}: FastF1 fetch failed")
                    continue

                corners, total_m = data
                if not corners:
                    logger.warning("Skipping circuit=%r — zero corners returned", circuit_name)
                    result.circuits_skipped += 1
                    result.errors.append(f"{circuit_name}: zero corners returned")
                    continue

                values: dict[str, Any] = {
                    "name": circuit_name,
                    "corners": corners,
                }
                # Back-fill length_km from marshal sector data if not already set.
                if length_km is None and total_m is not None:
                    values["length_km"] = round(total_m / 1000, 3)

                upsert_one(
                    db,
                    Circuit.__table__,
                    values=values,
                    conflict_cols=["name"],
                    update_cols=[k for k in values if k != "name"],
                )
                logger.info(
                    "Stored %d corners for circuit=%r", len(corners), circuit_name
                )
                result.circuits_updated += 1

            db.commit()
        except Exception:
            db.rollback()
            raise

    return result
