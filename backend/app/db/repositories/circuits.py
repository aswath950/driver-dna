"""Circuit queries."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Circuit, Event
from app.db.models import Session as SessionRow


async def get_for_session(db: AsyncSession, session_id: int) -> Circuit | None:
    """Return the Circuit row that the given session was held at, joining
    ``Session → Event → Circuit``. ``None`` if the session doesn't exist.
    """
    stmt = (
        select(Circuit)
        .join(Event, Event.circuit_id == Circuit.id)
        .join(SessionRow, SessionRow.event_id == Event.id)
        .where(SessionRow.id == session_id)
    )
    return (await db.scalars(stmt)).first()


async def get_by_name(db: AsyncSession, name: str) -> Circuit | None:
    stmt = select(Circuit).where(Circuit.name == name)
    return (await db.scalars(stmt)).first()
