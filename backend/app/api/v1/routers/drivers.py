from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.v1.schemas.driver import DriverOut, DriverStatsOut
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
from app.db.repositories import drivers as drivers_repo

router = APIRouter(tags=["drivers"])


@router.get(
    "/drivers",
    response_model=Page[DriverOut],
    summary="List drivers, filterable by season and team.",
)
async def list_drivers(
    db: DB,
    season: int | None = Query(default=None, description="Filter to drivers active in this season year."),
    team: str | None = Query(default=None, description="Filter to drivers who raced for this team."),
    cursor: CursorParam = None,
    limit: LimitParam = DEFAULT_LIMIT,
) -> Page[DriverOut]:
    decoded = decode_cursor(cursor)
    c_code, c_id = (decoded[0], decoded[1]) if decoded else (None, None)
    rows = await drivers_repo.list_drivers(
        db,
        season_year=season,
        team_name=team,
        cursor_code=c_code,
        cursor_id=c_id,
        limit=limit,
    )
    has_more = len(rows) > limit
    visible = rows[:limit]
    data = [DriverOut.model_validate(d) for d in visible]
    if has_more and visible:
        last = visible[-1]
        return build_page(rows=data, limit=limit, next_sort_key=last.code, next_pk=last.id)
    return build_page(rows=data, limit=limit)


@router.get(
    "/drivers/{driver_id}/stats",
    response_model=DriverStatsOut,
    responses={404: {"model": ErrorEnvelope}},
    summary="Aggregate stats for one driver in one season.",
)
async def get_driver_stats(
    driver_id: int,
    db: DB,
    season: int = Query(..., description="Season year (required)."),
) -> DriverStatsOut:
    if await drivers_repo.get_driver(db, driver_id) is None:
        raise NotFoundError("driver", driver_id)
    stats = await drivers_repo.get_driver_stats(
        db, driver_id=driver_id, season_year=season
    )
    if stats is None:
        raise NotFoundError("driver_stats", f"driver={driver_id} season={season}")
    return DriverStatsOut.model_validate(stats)
