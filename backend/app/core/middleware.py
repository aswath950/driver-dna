"""Request-scoped middleware: request IDs, API version header, structured logs."""

from __future__ import annotations

import time

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from ulid import ULID

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Read ``X-Request-ID`` from the client or mint a ULID. Bind it to the
    structlog contextvar so every log line in the request scope carries it,
    and echo on the response header.
    """

    header_name = "X-Request-ID"

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        rid = request.headers.get(self.header_name) or str(ULID())
        request.state.request_id = rid
        structlog.contextvars.bind_contextvars(request_id=rid)
        try:
            response: Response = await call_next(request)
        finally:
            structlog.contextvars.unbind_contextvars("request_id")
        response.headers[self.header_name] = rid
        return response


class APIVersionMiddleware(BaseHTTPMiddleware):
    """Echo ``API-Version: <n>`` on any response under ``/api/vN/*``."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        response: Response = await call_next(request)
        path = request.url.path
        if path.startswith("/api/v"):
            # /api/v1/... -> "1"
            try:
                version = path.split("/")[2][1:]  # drop the "v"
                if version.isdigit():
                    response.headers["API-Version"] = version
            except IndexError:
                pass
        return response


class AccessLogMiddleware(BaseHTTPMiddleware):
    """One JSON log line per request: method, path, status, latency_ms."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        t0 = time.perf_counter()
        response: Response = await call_next(request)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            "http.request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_ms=latency_ms,
        )
        return response


def register_middleware(app) -> None:  # type: ignore[no-untyped-def]
    """Order matters: outermost middleware first (Starlette wraps in reverse
    order of ``add_middleware`` calls).

    Effective call order (request → response):
        AccessLog → APIVersion → UserSession → RequestID → route → ...
    """
    from app.core.sessions import UserSessionMiddleware

    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(APIVersionMiddleware)
    app.add_middleware(UserSessionMiddleware)
    app.add_middleware(RequestIDMiddleware)


__all__ = [
    "AccessLogMiddleware",
    "APIVersionMiddleware",
    "RequestIDMiddleware",
    "register_middleware",
    "settings",
]
