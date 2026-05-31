"""Season queries."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Season


async def list_seasons(
    db: AsyncSession,
    *,
    cursor_year: int | None = None,
    cursor_id: int | None = None,
    limit: int,
) -> list[Season]:
    """List seasons newest-first. Cursor key = year (DESC), tie-break on id."""
    stmt = select(Season).order_by(Season.year.desc(), Season.id.asc())
    if cursor_year is not None and cursor_id is not None:
        stmt = stmt.where(
            (Season.year < cursor_year)
            | ((Season.year == cursor_year) & (Season.id > cursor_id))
        )
    stmt = stmt.limit(limit + 1)
    return list((await db.scalars(stmt)).all())


async def get_season_by_year(db: AsyncSession, year: int) -> Season | None:
    return await db.scalar(select(Season).where(Season.year == year))
