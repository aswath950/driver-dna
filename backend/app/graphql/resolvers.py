"""GraphQL resolvers — thin wrappers around the Phase 6 repositories.

No SQL is written here; everything routes through ``app.db.repositories``
so REST and GraphQL share identical query implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import strawberry
from sqlalchemy.ext.asyncio import AsyncSession
from strawberry.types import Info

from app.db.models import Driver as DriverModel
from app.db.models import Team as TeamModel
from app.db.repositories import drivers as drivers_repo
from app.db.repositories import events as events_repo
from app.db.repositories import laps as laps_repo
from app.db.repositories import results as results_repo
from app.db.repositories import seasons as seasons_repo
from app.db.repositories import sessions as sessions_repo
from app.db.repositories import standings as standings_repo
from app.graphql import types as gt

if TYPE_CHECKING:
    from app.graphql.loaders import Loaders


def _db(info: Info) -> AsyncSession:
    return info.context["db"]


def _loaders(info: Info) -> Loaders:
    return info.context["loaders"]


# ---------------------------------------------------------------------------
# Model → GraphQL converters
# ---------------------------------------------------------------------------


def _team_to_gql(t: TeamModel | None) -> gt.Team | None:
    if t is None:
        return None
    return gt.Team(id=strawberry.ID(str(t.id)), name=t.name, colorHex=t.color_hex)


def _driver_to_gql(d: DriverModel | None) -> gt.Driver | None:
    if d is None:
        return None
    team = _team_to_gql(getattr(d, "current_team", None))
    return gt.Driver(
        id=strawberry.ID(str(d.id)),
        code=d.code,
        fullName=d.full_name,
        nationality=d.nationality,
        currentTeam=team,
    )


def _season_to_gql(s) -> gt.Season:
    return gt.Season(id=strawberry.ID(str(s.id)), year=s.year)


def _event_to_gql(e) -> gt.Event:
    return gt.Event(
        id=strawberry.ID(str(e.id)),
        seasonId=strawberry.ID(str(e.season_id)),
        circuitId=strawberry.ID(str(e.circuit_id)),
        round=e.round,
        name=e.name,
        startDate=e.start_date,
    )


def _session_to_gql(s) -> gt.Session:
    return gt.Session(
        id=strawberry.ID(str(s.id)),
        eventId=strawberry.ID(str(s.event_id)),
        type=gt.SessionTypeGQL(s.type.value),
        dateStart=s.date_start,
        openf1SessionKey=s.openf1_session_key,
    )


def _lap_to_gql(l) -> gt.Lap:
    return gt.Lap(
        id=strawberry.ID(str(l.id)),
        sessionId=strawberry.ID(str(l.session_id)),
        driverId=strawberry.ID(str(l.driver_id)),
        lapNumber=l.lap_number,
        lapTimeMs=l.lap_time_ms,
        sector1Ms=l.sector1_ms,
        sector2Ms=l.sector2_ms,
        sector3Ms=l.sector3_ms,
        compound=gt.CompoundGQL(l.compound.value),
        tyreLife=l.tyre_life,
        isPitOut=l.is_pit_out,
        isPitIn=l.is_pit_in,
    )


# ---------------------------------------------------------------------------
# Query root
# ---------------------------------------------------------------------------


@strawberry.type
class Query:
    # --- seasons ---

    @strawberry.field
    async def season(self, info: Info, year: int) -> gt.Season | None:
        s = await seasons_repo.get_season_by_year(_db(info), year)
        return _season_to_gql(s) if s else None

    @strawberry.field
    async def seasons(self, info: Info, first: int = 20) -> list[gt.Season]:
        rows = await seasons_repo.list_seasons(
            _db(info), cursor_year=None, cursor_id=None, limit=first
        )
        return [_season_to_gql(s) for s in rows[:first]]

    # --- events ---

    @strawberry.field
    async def event(self, info: Info, id: strawberry.ID) -> gt.Event | None:
        e = await events_repo.get_event(_db(info), int(id))
        return _event_to_gql(e) if e else None

    @strawberry.field
    async def events(
        self, info: Info, seasonYear: int, first: int = 50
    ) -> list[gt.Event]:
        rows = await events_repo.list_events_for_season_year(
            _db(info),
            season_year=seasonYear,
            cursor_round=None,
            cursor_id=None,
            limit=first,
        )
        return [_event_to_gql(e) for e in rows[:first]]

    # --- sessions ---

    @strawberry.field
    async def session(self, info: Info, id: strawberry.ID) -> gt.Session | None:
        s = await sessions_repo.get_session(_db(info), int(id))
        return _session_to_gql(s) if s else None

    @strawberry.field
    async def sessionsForEvent(
        self, info: Info, eventId: strawberry.ID
    ) -> list[gt.Session]:
        rows = await sessions_repo.list_sessions_for_event(
            _db(info), event_id=int(eventId)
        )
        return [_session_to_gql(s) for s in rows]

    @strawberry.field
    async def sessionResults(
        self, info: Info, sessionId: strawberry.ID
    ) -> list[gt.RaceResult]:
        rows = await results_repo.get_leaderboard(
            _db(info), session_id=int(sessionId)
        )
        out: list[gt.RaceResult] = []
        for rr, driver, team in rows:
            out.append(
                gt.RaceResult(
                    sessionId=strawberry.ID(str(rr.session_id)),
                    position=rr.position,
                    grid=rr.grid,
                    points=float(rr.points),
                    status=rr.status,
                    fastestLapMs=rr.fastest_lap_ms,
                    driver=_driver_to_gql(driver),  # type: ignore[arg-type]
                    team=_team_to_gql(team),        # type: ignore[arg-type]
                )
            )
        return out

    @strawberry.field
    async def sessionLaps(
        self,
        info: Info,
        sessionId: strawberry.ID,
        driverId: strawberry.ID | None = None,
        fromLap: int | None = None,
        toLap: int | None = None,
        first: int = 100,
    ) -> list[gt.Lap]:
        rows = await laps_repo.list_laps(
            _db(info),
            session_id=int(sessionId),
            driver_id=int(driverId) if driverId else None,
            from_lap=fromLap,
            to_lap=toLap,
            cursor_lap=None,
            cursor_id=None,
            limit=first,
        )
        return [_lap_to_gql(l) for l in rows[:first]]

    # --- drivers ---

    @strawberry.field
    async def driver(self, info: Info, id: strawberry.ID) -> gt.Driver | None:
        # Goes through the DataLoader so repeated lookups in one request batch.
        d = await _loaders(info).drivers.load(int(id))
        if d is None:
            return None
        # Resolve current_team via loader too — single batched fetch
        # across all drivers in the request.
        team = None
        if d.current_team_id is not None:
            team_model = await _loaders(info).teams.load(d.current_team_id)
            team = _team_to_gql(team_model)
        return gt.Driver(
            id=strawberry.ID(str(d.id)),
            code=d.code,
            fullName=d.full_name,
            nationality=d.nationality,
            currentTeam=team,
        )

    @strawberry.field
    async def drivers(
        self,
        info: Info,
        season: int | None = None,
        team: str | None = None,
        first: int = 50,
    ) -> list[gt.Driver]:
        rows = await drivers_repo.list_drivers(
            _db(info),
            season_year=season,
            team_name=team,
            cursor_code=None,
            cursor_id=None,
            limit=first,
        )
        return [_driver_to_gql(d) for d in rows[:first]]  # type: ignore[misc]

    @strawberry.field
    async def driverStats(
        self,
        info: Info,
        driverId: strawberry.ID,
        season: int,
    ) -> gt.DriverStats | None:
        stats = await drivers_repo.get_driver_stats(
            _db(info), driver_id=int(driverId), season_year=season
        )
        if stats is None:
            return None
        return gt.DriverStats(
            driverId=strawberry.ID(str(stats.driver_id)),
            seasonId=strawberry.ID(str(stats.season_id)),
            wins=stats.wins,
            podiums=stats.podiums,
            poles=stats.poles,
            dnfs=stats.dnfs,
            points=float(stats.points),
            avgFinish=float(stats.avg_finish) if stats.avg_finish is not None else None,
        )

    # --- standings ---

    @strawberry.field
    async def standings(
        self, info: Info, season: int
    ) -> list[gt.StandingRow]:
        rows = await standings_repo.get_standings(_db(info), season_year=season)
        out: list[gt.StandingRow] = []
        for i, (stats, driver) in enumerate(rows, start=1):
            out.append(
                gt.StandingRow(
                    position=i,
                    driver=_driver_to_gql(driver),  # type: ignore[arg-type]
                    points=float(stats.points),
                    wins=stats.wins,
                    podiums=stats.podiums,
                )
            )
        return out
