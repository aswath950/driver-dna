"""Seed official corner data for all circuits from FastF1.

For each circuit in the database that hosts at least one event, resolves the
matching FastF1 race weekend and stores the official turn list (number,
letter, distance from S/F line in metres) in ``circuits.corners``.  Also
stores ``circuits.length_km`` from the same reference lap so corner distance
fractions are normalised consistently at runtime.

FastF1 sessions are addressed by *sequential* round number, but hydrated
events store the OpenF1 ``meeting_key`` in ``events.round`` — the two never
match.  The round is therefore resolved by exact case-insensitive name match
between the DB event name and FastF1's event schedule (the same name-based
join the web client uses for the GP dropdown).

Candidate (year, event_name) pairs come from each circuit's own events,
newest first, so a circuit that only hosts a 2026 event is still reachable.
Future-dated weekends are skipped (corner data only exists after the event),
and earlier seasons are tried as fallbacks since corner layouts are stable
year over year.

This is idempotent — re-running the command overwrites existing corner data
with the latest FastF1 values.

CLI usage:
    python -m app.etl seed-circuit-corners
    python -m app.etl seed-circuit-corners --year 2024   # prefer 2024 data
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fastf1
import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Circuit, Event, Season
from app.etl.circuit_aliases import canonical_circuit_name
from app.etl.upserts import upsert_one

logger = logging.getLogger(__name__)

# Events outside this window are sentinel/test rows (e.g. year 1899 / 2097).
_MIN_YEAR = 1950

# How many earlier seasons to try when a circuit's own event years all fail.
_FALLBACK_YEARS = 2


@dataclass
class SeedCornersResult:
    year: int | None
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

    Laps and telemetry must be loaded: FastF1 computes each marker's
    ``Distance`` from the S/F line by fitting the marker's x/y position
    against a reference lap's position telemetry
    (``CircuitInfo.add_marker_distance``).  Returns None on any error.
    """
    try:
        session = fastf1.get_session(year, round_num, "R")
        session.load(telemetry=True, laps=True, weather=False, messages=False)
        info = session.get_circuit_info()

        corners = [
            {
                "number": int(row["Number"]),
                "letter": str(row["Letter"]).strip(),
                "distance_m": float(row["Distance"]),
            }
            for _, row in info.corners.iterrows()
            # Distance is NaN when telemetry fitting failed for this event.
            if not pd.isna(row["Distance"])
        ]

        # Total lap length in metres.  Corner Distance values are sampled
        # from the reference lap's telemetry Distance channel, so the max of
        # that same channel is the consistent normaliser for apex fractions.
        # Marshal sectors are the fallback (they end slightly short of a lap).
        total_m: float | None = None
        try:
            ref_tel = session.laps.pick_fastest().get_telemetry()
            max_dist = ref_tel["Distance"].max()
            if not pd.isna(max_dist):
                total_m = float(max_dist)
        except Exception:
            pass
        if total_m is None and not info.marshal_sectors.empty:
            max_dist = info.marshal_sectors["Distance"].max()
            if not pd.isna(max_dist):
                total_m = float(max_dist)

        return corners, total_m

    except Exception as exc:
        logger.warning(
            "FastF1 fetch failed year=%d round=%d: %s", year, round_num, exc
        )
        return None


def _get_schedule_map(
    year: int,
    cache: dict[int, dict[str, tuple[int, pd.Timestamp | None]] | None],
) -> dict[str, tuple[int, pd.Timestamp | None]] | None:
    """Return {event_name_lower: (round_number, event_date)} for a season.

    Loaded at most once per year via ``cache``; None is cached on failure so
    a bad year isn't re-fetched for every circuit.
    """
    if year in cache:
        return cache[year]
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        mapping: dict[str, tuple[int, pd.Timestamp | None]] = {}
        for _, row in schedule.iterrows():
            event_date = row.get("EventDate")
            if pd.isna(event_date):
                event_date = None
            mapping[str(row["EventName"]).strip().lower()] = (
                int(row["RoundNumber"]),
                event_date,
            )
        cache[year] = mapping
    except Exception as exc:
        logger.warning("FastF1 schedule fetch failed year=%d: %s", year, exc)
        cache[year] = None
    return cache[year]


def _enable_cache() -> None:
    """Point FastF1 at a stable cache dir so re-runs don't re-download."""
    cache_dir = Path(tempfile.gettempdir()) / "fastf1_etl_cache"
    cache_dir.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))


def _build_candidates(
    event_pairs: list[tuple[int, str]],
    preferred_year: int | None = None,
) -> list[tuple[int, str]]:
    """Order (year, event_name) lookup candidates for one circuit.

    ``event_pairs`` must be newest-first.  Earlier seasons of the newest
    event name are appended as last-resort fallbacks — corner layouts rarely
    change between years.  A preferred year, when given, is tried first.
    """
    newest_year, newest_name = event_pairs[0]
    candidates = list(event_pairs)
    for back in range(1, _FALLBACK_YEARS + 1):
        pair = (newest_year - back, newest_name)
        if pair not in candidates:
            candidates.append(pair)
    if preferred_year is not None:
        candidates = [(preferred_year, name) for _, name in event_pairs] + candidates
    return candidates


def _store_circuit_corners(
    db: Session,
    circuit_name: str,
    corners: list[dict],
    total_m: float | None,
) -> None:
    """Upsert corners (and the paired length_km) onto a circuit row.

    length_km divides distance_m into apex fractions at runtime, so it must
    always be stored from the same FastF1 lap that produced the corner
    distances — never a stale value.
    """
    values: dict[str, Any] = {
        "name": circuit_name,
        "corners": corners,
    }
    if total_m is not None:
        values["length_km"] = round(total_m / 1000, 3)
    upsert_one(
        db,
        Circuit.__table__,
        values=values,
        conflict_cols=["name"],
        update_cols=[k for k in values if k != "name"],
    )


def _resolve_and_fetch(
    candidates: list[tuple[int, str]],
    schedule_cache: dict[int, dict[str, tuple[int, pd.Timestamp | None]] | None],
) -> tuple[int, list[dict], float | None] | None:
    """Try candidates in order; return (year, corners, total_m) for the first hit.

    A candidate is skipped when its year's schedule is unavailable, the event
    name has no exact match in that schedule, the event hasn't happened yet,
    or FastF1 fails to serve the circuit info.
    """
    now = pd.Timestamp.now()
    for year, event_name in candidates:
        schedule = _get_schedule_map(year, schedule_cache)
        if schedule is None:
            continue
        # OpenF1 event names can differ from FastF1's canonical schedule name
        # (e.g. "Barcelona Grand Prix" → "Spanish Grand Prix"); normalise before
        # matching so aliased weekends still resolve.
        lookup_name = canonical_circuit_name(event_name) or event_name
        match = schedule.get(lookup_name.strip().lower())
        if match is None:
            logger.info(
                "No schedule match for event=%r (lookup=%r) year=%d — trying next candidate",
                event_name, lookup_name, year,
            )
            continue
        round_num, event_date = match
        if event_date is not None and event_date > now:
            logger.info(
                "Event %r year=%d is in the future — trying next candidate",
                event_name, year,
            )
            continue
        data = _fetch_circuit_info(year, round_num)
        if data is None:
            continue
        corners, total_m = data
        if not corners:
            logger.warning(
                "Zero corners returned for event=%r year=%d round=%d",
                event_name, year, round_num,
            )
            continue
        return year, corners, total_m
    return None


def seed_for_event(db: Session, event_id: int) -> bool:
    """Targeted corner seed for one hydrated event's circuit.

    Used by the hydrate ETL so a newly downloaded GP gets corner data
    without re-running the full batch.  No-op when the circuit already has
    corners.  Commits on success; FastF1 failures are logged and swallowed
    (corner data is an enhancement, never a reason to fail a hydrate).

    Returns True when corner data was stored.
    """
    row = db.execute(
        select(Circuit.id, Circuit.name, Circuit.corners)
        .join(Event, Event.circuit_id == Circuit.id)
        .where(Event.id == event_id)
    ).first()
    if row is None or row.corners is not None:
        return False
    circuit_id, circuit_name = int(row.id), str(row.name)

    current_year = pd.Timestamp.now().year
    pairs = db.execute(
        select(Season.year, Event.name)
        .join(Event, Event.season_id == Season.id)
        .where(
            Event.circuit_id == circuit_id,
            Season.year.between(_MIN_YEAR, current_year),
        )
        .order_by(Season.year.desc())
    ).all()
    event_pairs: list[tuple[int, str]] = []
    for event_year, event_name in pairs:
        pair = (int(event_year), str(event_name))
        if pair not in event_pairs:
            event_pairs.append(pair)
    if not event_pairs:
        return False

    _enable_cache()
    resolved = _resolve_and_fetch(_build_candidates(event_pairs), {})
    if resolved is None:
        logger.warning(
            "Corner seed skipped for circuit=%r — no FastF1 match", circuit_name
        )
        return False

    resolved_year, corners, total_m = resolved
    _store_circuit_corners(db, circuit_name, corners, total_m)
    db.commit()
    logger.info(
        "Stored %d corners for circuit=%r (FastF1 season %d)",
        len(corners), circuit_name, resolved_year,
    )
    return True


def run(*, year: int | None = None) -> SeedCornersResult:
    """Fetch and store corner data for every circuit that hosts an event.

    Args:
        year: Optional preferred season — tried first for each circuit before
              falling back to the circuit's own event years (newest first).
    """
    _enable_cache()

    engine = create_engine(settings.DATABASE_URL_SYNC, future=True)
    result = SeedCornersResult(year=year, circuits_updated=0, circuits_skipped=0)
    current_year = pd.Timestamp.now().year

    with Session(engine) as db:
        rows = db.execute(
            select(Circuit.id, Circuit.name, Event.name, Season.year)
            .join(Event, Event.circuit_id == Circuit.id)
            .join(Season, Season.id == Event.season_id)
            .where(Season.year.between(_MIN_YEAR, current_year))
            .order_by(Circuit.id, Season.year.desc())
        ).all()

        # circuit_id → (circuit_name, [(year, event_name), ...] newest first)
        circuits: dict[int, tuple[str, list[tuple[int, str]]]] = {}
        for circuit_id, circuit_name, event_name, event_year in rows:
            entry = circuits.setdefault(circuit_id, (circuit_name, []))
            pair = (int(event_year), str(event_name))
            if pair not in entry[1]:
                entry[1].append(pair)

        schedule_cache: dict[int, dict[str, tuple[int, pd.Timestamp | None]] | None] = {}

        try:
            for circuit_name, event_pairs in circuits.values():
                candidates = _build_candidates(event_pairs, preferred_year=year)
                logger.info(
                    "Fetching corners  circuit=%r  candidates=%s",
                    circuit_name, candidates[:4],
                )
                resolved = _resolve_and_fetch(candidates, schedule_cache)
                if resolved is None:
                    logger.warning(
                        "Skipping circuit=%r — no FastF1 match for any candidate event",
                        circuit_name,
                    )
                    result.circuits_skipped += 1
                    result.errors.append(
                        f"{circuit_name}: no FastF1 match for any candidate event"
                    )
                    continue

                resolved_year, corners, total_m = resolved
                _store_circuit_corners(db, circuit_name, corners, total_m)
                logger.info(
                    "Stored %d corners for circuit=%r (FastF1 season %d)",
                    len(corners), circuit_name, resolved_year,
                )
                result.circuits_updated += 1

            db.commit()
        except Exception:
            db.rollback()
            raise

    return result
