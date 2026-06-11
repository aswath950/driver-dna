"""RFC 7807 Problem Details for HTTP APIs.

All API errors return the same envelope, with `request_id` added so users
can quote it back to us. Custom exception classes raised from routers and
services map to specific `type` URIs in the registry below.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError, NoResultFound
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger

logger = get_logger(__name__)

ERROR_BASE_URI = "https://driver-dna.dev/errors"


class ErrorEnvelope(BaseModel):
    """RFC 7807 Problem Details, plus our `request_id` extension."""

    type: str = Field(examples=[f"{ERROR_BASE_URI}/not_found"])
    title: str = Field(examples=["Resource not found"])
    status: int = Field(examples=[404])
    detail: str | None = Field(default=None, examples=["session 99 not found"])
    instance: str | None = Field(default=None, examples=["/api/v1/sessions/99"])
    request_id: str | None = Field(default=None, examples=["01HW6ABCDEF"])


# ---------------------------------------------------------------------------
# Domain exception hierarchy. Routers raise these; handlers below convert.
# ---------------------------------------------------------------------------


class APIError(Exception):
    """Base class for application errors that map to RFC 7807 envelopes."""

    status_code: int = 500
    error_type: str = "internal"
    title: str = "Internal server error"

    def __init__(self, detail: str | None = None) -> None:
        super().__init__(detail or self.title)
        self.detail = detail


class NotFoundError(APIError):
    status_code = 404
    error_type = "not_found"
    title = "Resource not found"

    def __init__(self, resource: str, identifier: Any) -> None:
        super().__init__(f"{resource} {identifier!r} not found")


class ConflictError(APIError):
    status_code = 409
    error_type = "conflict"
    title = "Conflict"


class BadRequestError(APIError):
    status_code = 400
    error_type = "bad_request"
    title = "Bad request"


class RateLimitedError(APIError):
    status_code = 429
    error_type = "rate_limited"
    title = "Too many requests"


class UpstreamError(APIError):
    status_code = 503
    error_type = "upstream_error"
    title = "Upstream service unavailable"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_envelope(
    *,
    status_code: int,
    error_type: str,
    title: str,
    detail: str | None,
    request: Request,
) -> ErrorEnvelope:
    return ErrorEnvelope(
        type=f"{ERROR_BASE_URI}/{error_type}",
        title=title,
        status=status_code,
        detail=detail,
        instance=request.url.path,
        request_id=getattr(request.state, "request_id", None),
    )


def _json_response(env: ErrorEnvelope) -> JSONResponse:
    return JSONResponse(
        status_code=env.status,
        content=env.model_dump(exclude_none=True),
        media_type="application/problem+json",
    )


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    logger.warning(
        "api.error",
        type=exc.error_type,
        status=exc.status_code,
        path=request.url.path,
        detail=exc.detail,
    )
    return _json_response(
        _build_envelope(
            status_code=exc.status_code,
            error_type=exc.error_type,
            title=exc.title,
            detail=exc.detail,
            request=request,
        )
    )


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    # FastAPI/Starlette HTTPException -> envelope.
    error_type, title = {
        400: ("bad_request", "Bad request"),
        401: ("unauthorized", "Unauthorized"),
        403: ("forbidden", "Forbidden"),
        404: ("not_found", "Resource not found"),
        405: ("method_not_allowed", "Method not allowed"),
        409: ("conflict", "Conflict"),
        429: ("rate_limited", "Too many requests"),
    }.get(exc.status_code, ("http_error", "HTTP error"))
    return _json_response(
        _build_envelope(
            status_code=exc.status_code,
            error_type=error_type,
            title=title,
            detail=str(exc.detail) if exc.detail else None,
            request=request,
        )
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return _json_response(
        _build_envelope(
            status_code=422,
            error_type="validation_error",
            title="Request validation failed",
            detail=str(exc.errors()),
            request=request,
        )
    )


async def integrity_error_handler(request: Request, exc: IntegrityError) -> JSONResponse:
    return _json_response(
        _build_envelope(
            status_code=409,
            error_type="conflict",
            title="Database constraint violation",
            detail=str(exc.orig) if exc.orig else None,
            request=request,
        )
    )


async def not_found_db_handler(request: Request, exc: NoResultFound) -> JSONResponse:
    return _json_response(
        _build_envelope(
            status_code=404,
            error_type="not_found",
            title="Resource not found",
            detail=str(exc) or None,
            request=request,
        )
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("api.unhandled", path=request.url.path)
    return _json_response(
        _build_envelope(
            status_code=500,
            error_type="internal",
            title="Internal server error",
            # Don't leak the raw exception message in prod responses.
            detail=None,
            request=request,
        )
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Wire all handlers into the FastAPI app."""
    app.add_exception_handler(APIError, api_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(IntegrityError, integrity_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(NoResultFound, not_found_db_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)
