from __future__ import annotations

from app.api.v1.schemas.common import ORMModel
from app.db.models import CompoundType


class LapOut(ORMModel):
    id: int
    session_id: int
    driver_id: int
    lap_number: int
    lap_time_ms: int | None = None
    sector1_ms: int | None = None
    sector2_ms: int | None = None
    sector3_ms: int | None = None
    compound: CompoundType
    tyre_life: int | None = None
    is_pit_out: bool
    is_pit_in: bool
