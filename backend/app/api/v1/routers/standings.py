from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.v1.schemas.driver import DriverOut
from app.api.v1.schemas.standing import StandingRowOut
from app.core.deps import DB
from app.db.repositories import standings as standings_repo

router = APIRouter(tags=["standings"])


@router.get(
    "/standings",
    response_model=list[StandingRowOut],
    summary="Driver standings for one season (ordered by points DESC).",
)
async def get_standings(
    db: DB,
    season: int = Query(..., description="Season year (required)."),
) -> list[StandingRowOut]:
    rows = await standings_repo.get_standings(db, season_year=season)
    out: list[StandingRowOut] = []
    for pos, (stats, driver) in enumerate(rows, start=1):
        out.append(
            StandingRowOut.model_validate({
                "position": pos,
                "driver": DriverOut.model_validate(driver),
                "points": stats.points,
                "wins": stats.wins,
                "podiums": stats.podiums,
            })
        )
    return out
