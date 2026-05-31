"""Season-standings queries."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.db.models import Driver, DriverStats, Season


async def get_standings(
    db: AsyncSession, *, season_year: int
) -> list[tuple[DriverStats, Driver]]:
    """Standings for one season, ordered by points DESC.

    Uses the ``ix_driver_stats_season_points`` index from migration 0002.
    Eager-loads ``Driver.current_team`` so the router can serialise without
    triggering an async lazy-load.
    """
    stmt = (
        select(DriverStats, Driver)
        .join(Season, Season.id == DriverStats.season_id)
        .join(Driver, Driver.id == DriverStats.driver_id)
        .where(Season.year == season_year)
        .order_by(DriverStats.points.desc(), Driver.code.asc())
        .options(joinedload(Driver.current_team))
    )
    rows = (await db.execute(stmt)).unique().all()
    return [(r[0], r[1]) for r in rows]
