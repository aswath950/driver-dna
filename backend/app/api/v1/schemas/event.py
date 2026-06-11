from __future__ import annotations

from datetime import date

from app.api.v1.schemas.common import ORMModel


class EventOut(ORMModel):
    id: int
    season_id: int
    circuit_id: int
    round: int
    name: str
    start_date: date | None = None
