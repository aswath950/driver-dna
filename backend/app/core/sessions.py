"""Anonymous-session middleware.

For every request we ensure a ``dna_sid`` cookie exists. If absent, we
mint a new ``user_sessions`` row and set the cookie on the response. If
present, we touch ``last_seen_at`` (best-effort) and stash the id on
``request.state.user_session_id`` so downstream code (LLM audit,
``/api/v1/me/*``, GraphQL) can attribute the request.

Paths we skip:
  - ``/healthz``, ``/openapi.json``, ``/docs``, ``/redoc`` — no need to
    mint a row for every health probe / spec fetch.
"""

from __future__ import annotations

import uuid

from sqlalchemy import update
from sqlalchemy.exc import SQLAlchemyError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.config import settings
from app.core.logging import get_logger
from app.db.models import UserSession
from app.db.session import AsyncSessionLocal

logger = get_logger(__name__)

COOKIE_NAME = "dna_sid"
COOKIE_MAX_AGE = 60 * 60 * 24 * 365  # 1 year

_SKIP_PATHS: tuple[str, ...] = (
    "/healthz",
    "/openapi.json",
    "/docs",
    "/redoc",
    "/favicon.ico",
)


def _parse_uuid(raw: str | None) -> uuid.UUID | None:
    if not raw:
        return None
    try:
        return uuid.UUID(raw)
    except (ValueError, TypeError):
        return None


def _cookie_attrs() -> dict:
    """Cookie attributes vary by environment.

    Prod (HTTPS): SameSite=None, Secure=True so cross-site Vercel→Railway
                  requests carry the cookie.
    Local (HTTP): SameSite=Lax, Secure=False so dev browsers accept it.
    """
    is_prod = settings.ENV not in ("local", "test")
    return {
        "httponly": True,
        "samesite": "none" if is_prod else "lax",
        "secure": is_prod,
        "max_age": COOKIE_MAX_AGE,
        "path": "/",
    }


class UserSessionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        if any(request.url.path.startswith(p) for p in _SKIP_PATHS):
            return await call_next(request)

        sid = _parse_uuid(request.cookies.get(COOKIE_NAME))
        is_new = False

        try:
            async with AsyncSessionLocal() as db:
                if sid is not None:
                    # Touch existing row; create if it's been GC'd.
                    res = await db.execute(
                        update(UserSession)
                        .where(UserSession.id == sid)
                        .values(last_seen_at=__import__("sqlalchemy").func.now())
                    )
                    if res.rowcount == 0:
                        # Cookie value points to a non-existent row — re-create.
                        sid = None

                if sid is None:
                    is_new = True
                    row = UserSession(
                        ua=request.headers.get("user-agent"),
                        locale=request.headers.get("accept-language"),
                    )
                    db.add(row)
                    await db.flush()
                    sid = row.id

                await db.commit()
        except SQLAlchemyError as e:  # pragma: no cover — best effort
            logger.warning("session.middleware.db_error", error=str(e))
            sid = sid or uuid.uuid4()

        request.state.user_session_id = sid

        response: Response = await call_next(request)
        if is_new:
            response.set_cookie(COOKIE_NAME, str(sid), **_cookie_attrs())
        return response


def get_user_session_id(request: Request) -> uuid.UUID | None:
    """FastAPI dep — returns the current session id from request.state."""
    return getattr(request.state, "user_session_id", None)
