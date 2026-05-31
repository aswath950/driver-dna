"""FastAPI dependency shims. Centralised so routers stay terse."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db


def get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "")


def get_user_session_id(request: Request) -> uuid.UUID | None:
    return getattr(request.state, "user_session_id", None)


RequestID = Annotated[str, Depends(get_request_id)]
DB = Annotated[AsyncSession, Depends(get_db)]
UserSessionID = Annotated[uuid.UUID | None, Depends(get_user_session_id)]
