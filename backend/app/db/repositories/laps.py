"""Lap-time queries."""

from __future__ import annotations

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import LapTime


async def fastest_lap_sectors(
    db: AsyncSession,
    *,
    session_id: int,
    driver_id: int,
) -> LapTime | None:
    """Return the fastest non-pit-out lap row with a recorded lap time, or None."""
    stmt = (
        select(LapTime)
        .where(
            and_(
                LapTime.session_id == session_id,
                LapTime.driver_id == driver_id,
                LapTime.is_pit_out == False,  # noqa: E712
                LapTime.lap_time_ms.is_not(None),
            )
        )
        .order_by(LapTime.lap_time_ms.asc())
        .limit(1)
    )
    return (await db.scalars(stmt)).first()


async def list_laps(
    db: AsyncSession,
    *,
    session_id: int,
    driver_id: int | None = None,
    from_lap: int | None = None,
    to_lap: int | None = None,
    cursor_lap: int | None = None,
    cursor_id: int | None = None,
    limit: int,
) -> list[LapTime]:
    """List lap_times for a session ordered by ``(lap_number, id)``.

    Cursor is a composite of ``(lap_number, lap_times.id)``; the WHERE clause
    keeps row order strictly monotonic across pages.

    Uses the unique index ``ux_lap_times_session_driver_lap`` when filtered
    by driver, otherwise the ``ix_lap_times_session_lap`` index from 0002.
    """
    clauses = [LapTime.session_id == session_id]
    if driver_id is not None:
        clauses.append(LapTime.driver_id == driver_id)
    if from_lap is not None:
        clauses.append(LapTime.lap_number >= from_lap)
    if to_lap is not None:
        clauses.append(LapTime.lap_number <= to_lap)
    if cursor_lap is not None and cursor_id is not None:
        clauses.append(
            (LapTime.lap_number > cursor_lap)
            | ((LapTime.lap_number == cursor_lap) & (LapTime.id > cursor_id))
        )

    stmt = (
        select(LapTime)
        .where(and_(*clauses))
        .order_by(LapTime.lap_number.asc(), LapTime.id.asc())
        .limit(limit + 1)
    )
    return list((await db.scalars(stmt)).all())
