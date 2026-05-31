from __future__ import annotations

from app.api.v1.schemas.common import ORMModel


class TeamOut(ORMModel):
    id: int
    name: str
    color_hex: str | None = None
