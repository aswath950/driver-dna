"""Driver + driver-stats queries."""

from __future__ import annotations

from sqlalchemy import and_, distinct, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.db.models import Driver, DriverStats, Event, Season, SessionDriver, Team
from app.db.models import Session as SessionRow


async def list_drivers(
    db: AsyncSession,
    *,
    season_year: int | None = None,
    team_name: str | None = None,
    cursor_code: str | None = None,
    cursor_id: int | None = None,
    limit: int,
) -> list[Driver]:
    """List drivers, optionally filtered to those who appear in a given
    season or who currently/historically drive for a named team.

    Ordering: ``code ASC, id ASC`` (alphabetic by 3-letter acronym).
    """
    stmt = select(Driver).options(joinedload(Driver.current_team))

    if season_year is not None:
        stmt = stmt.where(
            Driver.id.in_(
                select(distinct(SessionDriver.driver_id))
                .join(SessionRow, SessionRow.id == SessionDriver.session_id)
                .join(Event, Event.id == SessionRow.event_id)
                .join(Season, Season.id == Event.season_id)
                .where(Season.year == season_year)
            )
        )

    if team_name is not None:
        # Match drivers who ever raced for that team in any session.
        stmt = stmt.where(
            Driver.id.in_(
                select(distinct(SessionDriver.driver_id))
                .join(Team, Team.id == SessionDriver.team_id)
                .where(Team.name == team_name)
            )
        )

    if cursor_code is not None and cursor_id is not None:
        stmt = stmt.where(
            (Driver.code > cursor_code)
            | ((Driver.code == cursor_code) & (Driver.id > cursor_id))
        )

    stmt = stmt.order_by(Driver.code.asc(), Driver.id.asc()).limit(limit + 1)
    return list((await db.scalars(stmt)).unique().all())


async def get_driver(db: AsyncSession, driver_id: int) -> Driver | None:
    return await db.get(Driver, driver_id)


async def get_driver_stats(
    db: AsyncSession, *, driver_id: int, season_year: int
) -> DriverStats | None:
    stmt = (
        select(DriverStats)
        .join(Season, Season.id == DriverStats.season_id)
        .where(and_(DriverStats.driver_id == driver_id, Season.year == season_year))
    )
    return await db.scalar(stmt)
