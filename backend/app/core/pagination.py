"""Opaque-cursor pagination.

Cursors are base64url-encoded JSON ``{"k": <sort_key>, "id": <pk>}`` —
clients never construct them, only echo what the server returned. The
secondary tie-break on ``id`` keeps order stable when two rows share the
same sort key.
"""

from __future__ import annotations

import base64
import json
from typing import Annotated, Any, Generic, TypeVar

from fastapi import Query
from pydantic import BaseModel, Field

from app.core.errors import BadRequestError

T = TypeVar("T")

DEFAULT_LIMIT = 50
MAX_LIMIT = 200


class PageInfo(BaseModel):
    next_cursor: str | None = Field(default=None, examples=["eyJrIjogMTIsICJpZCI6IDQyfQ"])
    has_more: bool = False
    limit: int = Field(examples=[DEFAULT_LIMIT])


class Page(BaseModel, Generic[T]):
    """Envelope returned by every list endpoint."""

    data: list[T]
    page: PageInfo


# ---------------------------------------------------------------------------
# Cursor encode / decode
# ---------------------------------------------------------------------------


def encode_cursor(sort_key: Any, pk: int) -> str:
    """Pack ``(sort_key, pk)`` into an opaque base64url string.

    ``sort_key`` must be JSON-serialisable. Datetimes should be converted to
    ISO strings by the caller before passing in.
    """
    raw = json.dumps({"k": sort_key, "id": int(pk)}, separators=(",", ":"))
    return base64.urlsafe_b64encode(raw.encode("utf-8")).rstrip(b"=").decode("ascii")


def decode_cursor(cursor: str | None) -> tuple[Any, int] | None:
    """Reverse of :func:`encode_cursor`. Returns ``None`` for an empty cursor,
    raises ``BadRequestError`` for a malformed one.
    """
    if cursor is None or cursor == "":
        return None
    try:
        # Restore padding stripped by encode_cursor.
        padded = cursor + "=" * (-len(cursor) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
        obj = json.loads(raw)
        if not isinstance(obj, dict) or "k" not in obj or "id" not in obj:
            raise ValueError("cursor payload missing k/id")
        return obj["k"], int(obj["id"])
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as e:
        raise BadRequestError(f"invalid cursor: {e}") from e


# ---------------------------------------------------------------------------
# FastAPI query-param dependency
# ---------------------------------------------------------------------------

CursorParam = Annotated[str | None, Query(description="Opaque page cursor.")]
LimitParam = Annotated[
    int,
    Query(ge=1, le=MAX_LIMIT, description=f"Page size (max {MAX_LIMIT})."),
]


def build_page(
    *,
    rows: list[T],
    limit: int,
    next_sort_key: Any = None,
    next_pk: int | None = None,
) -> Page[T]:
    """Construct a Page envelope from a result set.

    Caller fetches ``limit + 1`` rows and passes the first ``limit`` here;
    the +1 indicates ``has_more``. If ``has_more`` is True, pass the +1
    row's sort key and pk into ``next_sort_key`` / ``next_pk``.
    """
    has_more = next_sort_key is not None and next_pk is not None
    next_cursor = (
        encode_cursor(next_sort_key, next_pk) if has_more else None
    )
    return Page[T](  # type: ignore[valid-type]
        data=rows,
        page=PageInfo(next_cursor=next_cursor, has_more=has_more, limit=limit),
    )
