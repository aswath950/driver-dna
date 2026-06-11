from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import select

from app.api.v1.schemas.lap import LapOut
from app.api.v1.schemas.race_result import RaceResultOut
from app.api.v1.schemas.session import SessionOut
from app.core.deps import DB
from app.core.errors import ErrorEnvelope, NotFoundError
from app.core.pagination import (
    DEFAULT_LIMIT,
    CursorParam,
    LimitParam,
    Page,
    build_page,
    decode_cursor,
)
from app.db.models import SessionDriver, Team
from app.db.repositories import laps as laps_repo
from app.db.repositories import results as results_repo
from app.db.repositories import sessions as sessions_repo

router = APIRouter(tags=["sessions"])


class SessionTeamOut(BaseModel):
    id: int
    name: str
    color_hex: str | None = None


@router.get(
    "/sessions/{session_id}",
    response_model=SessionOut,
    responses={404: {"model": ErrorEnvelope}},
    summary="Get one session by id.",
)
async def get_session(session_id: int, db: DB) -> SessionOut:
    s = await sessions_repo.get_session(db, session_id)
    if s is None:
        raise NotFoundError("session", session_id)
    return SessionOut.model_validate(s)


@router.get(
    "/sessions/{session_id}/results",
    response_model=list[RaceResultOut],
    responses={404: {"model": ErrorEnvelope}},
    summary="Leaderboard for one session (ordered by position).",
)
async def get_leaderboard(session_id: int, db: DB) -> list[RaceResultOut]:
    if await sessions_repo.get_session(db, session_id) is None:
        raise NotFoundError("session", session_id)
    rows = await results_repo.get_leaderboard(db, session_id=session_id)
    out: list[RaceResultOut] = []
    for rr, driver, team in rows:
        out.append(
            RaceResultOut.model_validate({
                "session_id": rr.session_id,
                "position": rr.position,
                "grid": rr.grid,
                "points": rr.points,
                "status": rr.status,
                "fastest_lap_ms": rr.fastest_lap_ms,
                "driver": driver,
                "team": team,
            })
        )
    return out


@router.get(
    "/sessions/{session_id}/laps",
    response_model=Page[LapOut],
    responses={404: {"model": ErrorEnvelope}},
    summary="Laps for a session — paginated, filterable by driver and lap range.",
)
async def list_laps(
    session_id: int,
    db: DB,
    driver_id: int | None = None,
    from_lap: int | None = None,
    to_lap: int | None = None,
    cursor: CursorParam = None,
    limit: LimitParam = DEFAULT_LIMIT,
) -> Page[LapOut]:
    if await sessions_repo.get_session(db, session_id) is None:
        raise NotFoundError("session", session_id)

    decoded = decode_cursor(cursor)
    c_lap, c_id = (decoded[0], decoded[1]) if decoded else (None, None)
    rows = await laps_repo.list_laps(
        db,
        session_id=session_id,
        driver_id=driver_id,
        from_lap=from_lap,
        to_lap=to_lap,
        cursor_lap=c_lap,
        cursor_id=c_id,
        limit=limit,
    )
    has_more = len(rows) > limit
    visible = rows[:limit]
    data = [LapOut.model_validate(l) for l in visible]
    if has_more and visible:
        last = visible[-1]
        return build_page(
            rows=data, limit=limit, next_sort_key=last.lap_number, next_pk=last.id
        )
    return build_page(rows=data, limit=limit)


@router.get(
    "/sessions/{session_id}/teams",
    response_model=list[SessionTeamOut],
    responses={404: {"model": ErrorEnvelope}},
    summary="Distinct teams participating in a session.",
)
async def list_session_teams(session_id: int, db: DB) -> list[SessionTeamOut]:
    if await sessions_repo.get_session(db, session_id) is None:
        raise NotFoundError("session", session_id)
    rows = (
        await db.execute(
            select(Team.id, Team.name, Team.color_hex)
            .join(SessionDriver, SessionDriver.team_id == Team.id)
            .where(SessionDriver.session_id == session_id)
            .distinct()
            .order_by(Team.name)
        )
    ).all()
    return [SessionTeamOut(id=r.id, name=r.name, color_hex=r.color_hex) for r in rows]
