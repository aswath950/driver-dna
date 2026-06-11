from __future__ import annotations

from app.api.v1.schemas.common import ORMModel


class SeasonOut(ORMModel):
    id: int
    year: int
