from __future__ import annotations

from datetime import datetime

from app.api.v1.schemas.common import ORMModel
from app.db.models import SessionType


class SessionOut(ORMModel):
    id: int
    event_id: int
    type: SessionType
    date_start: datetime | None = None
    openf1_session_key: int | None = None
