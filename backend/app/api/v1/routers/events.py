from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.schemas.session import SessionOut
from app.core.deps import DB
from app.core.errors import ErrorEnvelope, NotFoundError
from app.db.repositories import events as events_repo
from app.db.repositories import sessions as sessions_repo

router = APIRouter(tags=["events"])


@router.get(
    "/events/{event_id}/sessions",
    response_model=list[SessionOut],
    responses={404: {"model": ErrorEnvelope}},
    summary="List sessions for one event (chronological).",
)
async def list_sessions_for_event(event_id: int, db: DB) -> list[SessionOut]:
    if await events_repo.get_event(db, event_id) is None:
        raise NotFoundError("event", event_id)
    sessions = await sessions_repo.list_sessions_for_event(db, event_id=event_id)
    return [SessionOut.model_validate(s) for s in sessions]
