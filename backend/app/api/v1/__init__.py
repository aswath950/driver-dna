"""v1 REST router. Subrouters land in `routers/` from Phase 6 onward.

This module mounts the version namespace and ships a `/api/v1/_ping`
sentinel endpoint that exercises the full middleware + error stack.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.v1.routers import ai as ai_router
from app.api.v1.routers import analytics as analytics_router
from app.api.v1.routers import drivers as drivers_router
from app.api.v1.routers import events as events_router
from app.api.v1.routers import me as me_router
from app.api.v1.routers import seasons as seasons_router
from app.api.v1.routers import sessions as sessions_router
from app.api.v1.routers import standings as standings_router
from app.core.errors import BadRequestError, ErrorEnvelope, NotFoundError
from app.core.pagination import (
    CursorParam,
    LimitParam,
    Page,
    build_page,
    decode_cursor,
)

router = APIRouter(
    prefix="/api/v1",
    responses={
        400: {"model": ErrorEnvelope},
        404: {"model": ErrorEnvelope},
        422: {"model": ErrorEnvelope},
        500: {"model": ErrorEnvelope},
    },
)

router.include_router(seasons_router.router)
router.include_router(events_router.router)
router.include_router(sessions_router.router)
router.include_router(drivers_router.router)
router.include_router(standings_router.router)
router.include_router(analytics_router.router)
router.include_router(ai_router.router)
router.include_router(me_router.router)


@router.get("/_ping", summary="Sentinel endpoint for contract tests.")
async def ping() -> dict[str, str]:
    return {"pong": "ok"}


@router.get(
    "/_ping/raise/{kind}",
    summary="Force a typed error for envelope testing.",
    responses={404: {"model": ErrorEnvelope}, 400: {"model": ErrorEnvelope}},
)
async def raise_error(kind: str) -> None:
    if kind == "not_found":
        raise NotFoundError("ping", kind)
    if kind == "bad_request":
        raise BadRequestError("you asked for a bad request")
    raise BadRequestError(f"unknown kind: {kind}")


@router.get(
    "/_ping/page",
    summary="Sentinel paginated endpoint for cursor round-trip tests.",
    response_model=Page[int],
)
async def page_demo(
    cursor: CursorParam = None,
    limit: LimitParam = 5,
) -> Page[int]:
    """Returns a deterministic stream of integers paginated through cursors.

    The synthetic 'table' is the integers 1..100. Each row is also its own
    sort key; the secondary tie-break on id is therefore identical, which
    is fine for a sentinel.
    """
    decoded = decode_cursor(cursor)
    start = (decoded[1] + 1) if decoded else 1
    universe_end = 100
    # Fetch limit+1 to detect has_more cheaply.
    rows = list(range(start, min(start + limit + 1, universe_end + 1)))
    has_more = len(rows) > limit
    visible = rows[:limit]
    if has_more:
        # Cursor convention: encode the LAST visible row so the next page
        # starts strictly after it. Avoids drift from off-by-one bugs.
        last = visible[-1]
        return build_page(rows=visible, limit=limit, next_sort_key=last, next_pk=last)
    return build_page(rows=visible, limit=limit)


@router.get(
    "/_ping/echo-limit",
    summary="Exercises the LimitParam validator (1..200).",
)
async def echo_limit(limit: int = Query(default=10, ge=1, le=200)) -> dict[str, int]:
    return {"limit": limit}
