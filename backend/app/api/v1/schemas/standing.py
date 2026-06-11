from __future__ import annotations

from decimal import Decimal

from app.api.v1.schemas.common import ORMModel
from app.api.v1.schemas.driver import DriverOut


class StandingRowOut(ORMModel):
    position: int
    driver: DriverOut
    points: Decimal
    wins: int
    podiums: int
