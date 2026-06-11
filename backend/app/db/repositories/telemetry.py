"""Telemetry cache queries — async read/write for ``car_telemetry``."""

from __future__ import annotations

import json

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import CarTelemetry


async def get_lap(
    db: AsyncSession,
    session_id: int,
    driver_id: int,
    lap_number: int,
) -> CarTelemetry | None:
    """Return the cached telemetry row for a specific lap, or None on cache miss."""
    stmt = select(CarTelemetry).where(
        CarTelemetry.session_id == session_id,
        CarTelemetry.driver_id == driver_id,
        CarTelemetry.lap_number == lap_number,
    )
    return (await db.scalars(stmt)).first()


async def has_session_telemetry(db: AsyncSession, session_id: int) -> bool:
    """Return True if any telemetry rows exist for this session."""
    stmt = select(CarTelemetry.session_id).where(
        CarTelemetry.session_id == session_id
    ).limit(1)
    return (await db.scalars(stmt)).first() is not None


async def save_lap(
    db: AsyncSession,
    *,
    session_id: int,
    driver_id: int,
    lap_number: int,
    lap_duration: float | None,
    samples: dict,
) -> None:
    """Upsert one cached telemetry row (safe to call on cache miss in async context)."""
    stmt = (
        pg_insert(CarTelemetry.__table__)
        .values(
            session_id=session_id,
            driver_id=driver_id,
            lap_number=lap_number,
            lap_duration=lap_duration,
            samples=samples,
            fetched_at=func.now(),
        )
        .on_conflict_do_update(
            index_elements=["session_id", "driver_id", "lap_number"],
            set_={
                "lap_duration": lap_duration,
                "samples": samples,
                "fetched_at": func.now(),
            },
        )
    )
    await db.execute(stmt)
