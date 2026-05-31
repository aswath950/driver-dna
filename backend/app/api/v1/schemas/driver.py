from __future__ import annotations

from decimal import Decimal

from app.api.v1.schemas.common import ORMModel
from app.api.v1.schemas.team import TeamOut


class DriverOut(ORMModel):
    id: int
    code: str
    full_name: str
    nationality: str | None = None
    current_team: TeamOut | None = None


class DriverStatsOut(ORMModel):
    driver_id: int
    season_id: int
    wins: int
    podiums: int
    poles: int
    dnfs: int
    points: Decimal
    avg_finish: Decimal | None = None
