"""Event queries."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Event, Season


async def list_events_for_season_year(
    db: AsyncSession,
    *,
    season_year: int,
    cursor_round: int | None = None,
    cursor_id: int | None = None,
    limit: int,
) -> list[Event]:
    """List events for one season ordered by round ASC."""
    stmt = (
        select(Event)
        .join(Season, Season.id == Event.season_id)
        .where(Season.year == season_year)
        .order_by(Event.round.asc(), Event.id.asc())
    )
    if cursor_round is not None and cursor_id is not None:
        stmt = stmt.where(
            (Event.round > cursor_round)
            | ((Event.round == cursor_round) & (Event.id > cursor_id))
        )
    stmt = stmt.limit(limit + 1)
    return list((await db.scalars(stmt)).all())


async def get_event(db: AsyncSession, event_id: int) -> Event | None:
    return await db.get(Event, event_id)
