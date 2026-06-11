from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.schemas.event import EventOut
from app.api.v1.schemas.season import SeasonOut
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
from app.db.repositories import events as events_repo
from app.db.repositories import seasons as seasons_repo

router = APIRouter(tags=["seasons"])


@router.get(
    "/seasons",
    response_model=Page[SeasonOut],
    summary="List seasons (newest first).",
)
async def list_seasons(
    db: DB,
    cursor: CursorParam = None,
    limit: LimitParam = DEFAULT_LIMIT,
) -> Page[SeasonOut]:
    decoded = decode_cursor(cursor)
    cursor_year, cursor_id = (decoded[0], decoded[1]) if decoded else (None, None)
    rows = await seasons_repo.list_seasons(
        db, cursor_year=cursor_year, cursor_id=cursor_id, limit=limit
    )
    has_more = len(rows) > limit
    visible = rows[:limit]
    data = [SeasonOut.model_validate(s) for s in visible]
    if has_more and visible:
        last = visible[-1]
        return build_page(rows=data, limit=limit, next_sort_key=last.year, next_pk=last.id)
    return build_page(rows=data, limit=limit)


@router.get(
    "/seasons/{year}/events",
    response_model=Page[EventOut],
    responses={404: {"model": ErrorEnvelope}},
    summary="List events for one season (ordered by round).",
)
async def list_events_for_season(
    year: int,
    db: DB,
    cursor: CursorParam = None,
    limit: LimitParam = DEFAULT_LIMIT,
) -> Page[EventOut]:
    season = await seasons_repo.get_season_by_year(db, year)
    if season is None:
        raise NotFoundError("season", year)

    decoded = decode_cursor(cursor)
    c_round, c_id = (decoded[0], decoded[1]) if decoded else (None, None)
    rows = await events_repo.list_events_for_season_year(
        db, season_year=year, cursor_round=c_round, cursor_id=c_id, limit=limit
    )
    has_more = len(rows) > limit
    visible = rows[:limit]
    data = [EventOut.model_validate(e) for e in visible]
    if has_more and visible:
        last = visible[-1]
        return build_page(rows=data, limit=limit, next_sort_key=last.round, next_pk=last.id)
    return build_page(rows=data, limit=limit)
