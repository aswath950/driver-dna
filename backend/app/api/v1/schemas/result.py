from __future__ import annotations

from decimal import Decimal

from app.api.v1.schemas.common import ORMModel


class RaceResultOut(ORMModel):
    id: int
    session_id: int
    driver_id: int
    position: int | None = None
    grid: int | None = None
    points: Decimal
    status: str | None = None
    fastest_lap_ms: int | None = None
