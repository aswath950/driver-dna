"""Race-result (leaderboard) queries."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.db.models import Driver, RaceResult, SessionDriver, Team


async def get_leaderboard(
    db: AsyncSession, *, session_id: int
) -> list[tuple[RaceResult, Driver, Team]]:
    """Leaderboard with driver and team eagerly joined.

    Returns a list of (RaceResult, Driver, Team) tuples ordered by position
    (NULLs last so DNFs sort to the bottom).

    Uses the ``ix_race_results_session_pos`` index from migration 0002.
    """
    stmt = (
        select(RaceResult, Driver, Team)
        .join(Driver, Driver.id == RaceResult.driver_id)
        .join(
            SessionDriver,
            (SessionDriver.session_id == RaceResult.session_id)
            & (SessionDriver.driver_id == RaceResult.driver_id),
        )
        .join(Team, Team.id == SessionDriver.team_id)
        .where(RaceResult.session_id == session_id)
        .order_by(RaceResult.position.asc().nulls_last(), RaceResult.id.asc())
        .options(joinedload(Driver.current_team))
    )
    rows = (await db.execute(stmt)).unique().all()
    return [(r[0], r[1], r[2]) for r in rows]
