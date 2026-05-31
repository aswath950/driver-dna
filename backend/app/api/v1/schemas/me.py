"""Request + response schemas for /api/v1/me/*."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.api.v1.schemas.common import ORMModel
from app.db.models import AnalysisKind


class UserSessionOut(ORMModel):
    id: uuid.UUID
    created_at: datetime
    last_seen_at: datetime


class SavedAnalysisCreate(BaseModel):
    kind: AnalysisKind
    session_id: int | None = None
    payload: dict[str, Any] = Field(..., description="Free-form JSON-serialisable body.")


class SavedAnalysisOut(ORMModel):
    id: uuid.UUID
    user_session_id: uuid.UUID
    kind: AnalysisKind
    session_id: int | None = None
    payload: dict[str, Any]
    created_at: datetime
