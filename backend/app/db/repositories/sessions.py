"""Session queries."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Session as SessionRow


async def list_sessions_for_event(
    db: AsyncSession, *, event_id: int
) -> list[SessionRow]:
    """All sessions for one event, ordered chronologically.

    Not paginated — a race weekend has at most ~7 sessions.
    """
    stmt = (
        select(SessionRow)
        .where(SessionRow.event_id == event_id)
        .order_by(SessionRow.date_start.asc(), SessionRow.id.asc())
    )
    return list((await db.scalars(stmt)).all())


async def get_session(db: AsyncSession, session_id: int) -> SessionRow | None:
    return await db.get(SessionRow, session_id)
