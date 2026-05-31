"""/api/v1/me/* — anonymous-session views + saved analyses."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Response, status
from sqlalchemy import delete, func, select

from app.api.v1.schemas.me import (
    SavedAnalysisCreate,
    SavedAnalysisOut,
    UserSessionOut,
)
from app.core.deps import DB, UserSessionID
from app.core.errors import ConflictError, ErrorEnvelope, NotFoundError
from app.core.pagination import (
    DEFAULT_LIMIT,
    CursorParam,
    LimitParam,
    Page,
    build_page,
    decode_cursor,
)
from app.db.models import SavedAnalysis, UserSession

router = APIRouter(prefix="/me", tags=["me"])

SAVED_ANALYSES_CAP = 100


@router.get(
    "",
    response_model=UserSessionOut,
    responses={401: {"model": ErrorEnvelope}},
    summary="Current anonymous session — created on first request via cookie.",
)
async def me(db: DB, sid: UserSessionID) -> UserSessionOut:
    if sid is None:
        raise NotFoundError("user_session", "no cookie")
    row = await db.get(UserSession, sid)
    if row is None:
        raise NotFoundError("user_session", str(sid))
    return UserSessionOut.model_validate(row)


@router.post(
    "/saved-analyses",
    response_model=SavedAnalysisOut,
    status_code=status.HTTP_201_CREATED,
    responses={409: {"model": ErrorEnvelope}},
    summary="Save an analysis (radar / report-card / race-chat / xai / dna-match).",
)
async def create_saved_analysis(
    body: SavedAnalysisCreate, db: DB, sid: UserSessionID
) -> SavedAnalysisOut:
    if sid is None:
        raise NotFoundError("user_session", "no cookie")

    n = await db.scalar(
        select(func.count(SavedAnalysis.id)).where(
            SavedAnalysis.user_session_id == sid
        )
    )
    if int(n or 0) >= SAVED_ANALYSES_CAP:
        raise ConflictError(
            f"saved-analyses cap of {SAVED_ANALYSES_CAP} reached; delete older ones first"
        )

    row = SavedAnalysis(
        user_session_id=sid,
        kind=body.kind,
        session_id=body.session_id,
        payload=body.payload,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return SavedAnalysisOut.model_validate(row)


@router.get(
    "/saved-analyses",
    response_model=Page[SavedAnalysisOut],
    summary="List the current session's saved analyses (newest first).",
)
async def list_saved_analyses(
    db: DB,
    sid: UserSessionID,
    cursor: CursorParam = None,
    limit: LimitParam = DEFAULT_LIMIT,
) -> Page[SavedAnalysisOut]:
    if sid is None:
        return build_page(rows=[], limit=limit)

    decoded = decode_cursor(cursor)
    # Sort key is created_at (ISO string for stable cursor encoding); tie-break on id.
    stmt = select(SavedAnalysis).where(SavedAnalysis.user_session_id == sid)
    if decoded is not None:
        cursor_ts_str, cursor_id = decoded
        # cursor encodes (last_created_at_iso, last_id) — fetch strictly older.
        from datetime import datetime
        cursor_ts = datetime.fromisoformat(cursor_ts_str)
        stmt = stmt.where(
            (SavedAnalysis.created_at < cursor_ts)
            | (
                (SavedAnalysis.created_at == cursor_ts)
                & (SavedAnalysis.id < uuid.UUID(str(cursor_id)))
            )
        )
    stmt = stmt.order_by(SavedAnalysis.created_at.desc(), SavedAnalysis.id.desc()).limit(
        limit + 1
    )
    rows = list((await db.scalars(stmt)).all())
    has_more = len(rows) > limit
    visible = rows[:limit]
    data = [SavedAnalysisOut.model_validate(r) for r in visible]
    if has_more and visible:
        last = visible[-1]
        return build_page(
            rows=data,
            limit=limit,
            next_sort_key=last.created_at.isoformat(),
            next_pk=int(last.id.int),  # cursor needs an int; embed UUID as int
        )
    return build_page(rows=data, limit=limit)


@router.delete(
    "/saved-analyses/{analysis_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorEnvelope}},
    summary="Delete a saved analysis owned by the current session.",
)
async def delete_saved_analysis(
    analysis_id: uuid.UUID, db: DB, sid: UserSessionID
) -> Response:
    if sid is None:
        raise NotFoundError("saved_analysis", str(analysis_id))
    res = await db.execute(
        delete(SavedAnalysis).where(
            SavedAnalysis.id == analysis_id,
            SavedAnalysis.user_session_id == sid,  # ownership check
        )
    )
    if res.rowcount == 0:
        raise NotFoundError("saved_analysis", str(analysis_id))
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
