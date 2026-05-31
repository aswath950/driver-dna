from __future__ import annotations

from decimal import Decimal

from app.api.v1.schemas.common import ORMModel
from app.api.v1.schemas.driver import DriverOut
from app.api.v1.schemas.team import TeamOut


class RaceResultOut(ORMModel):
    """Leaderboard row — embeds driver + team for one-call rendering."""

    session_id: int
    position: int | None = None
    grid: int | None = None
    points: Decimal
    status: str | None = None
    fastest_lap_ms: int | None = None
    driver: DriverOut
    team: TeamOut
