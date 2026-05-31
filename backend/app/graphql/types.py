"""Strawberry types mirroring the v1 REST schemas.

Field names are GraphQL-idiomatic camelCase (Strawberry handles the
snake↔camel translation via ``strawberry.field(name=...)`` where needed).
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum

import strawberry

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


@strawberry.enum
class SessionTypeGQL(Enum):
    FP1 = "FP1"
    FP2 = "FP2"
    FP3 = "FP3"
    Q = "Q"
    SQ = "SQ"
    S = "S"
    R = "R"


@strawberry.enum
class CompoundGQL(Enum):
    SOFT = "SOFT"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    INTERMEDIATE = "INTERMEDIATE"
    WET = "WET"
    UNKNOWN = "UNKNOWN"


@strawberry.enum
class ChannelGQL(Enum):
    SPEED = "SPEED"
    THROTTLE = "THROTTLE"
    BRAKE = "BRAKE"


# ---------------------------------------------------------------------------
# Object types
# ---------------------------------------------------------------------------


@strawberry.type
class Team:
    id: strawberry.ID
    name: str
    colorHex: str | None = None


@strawberry.type
class Driver:
    id: strawberry.ID
    code: str
    fullName: str
    nationality: str | None = None
    currentTeam: Team | None = None


@strawberry.type
class DriverStats:
    driverId: strawberry.ID
    seasonId: strawberry.ID
    wins: int
    podiums: int
    poles: int
    dnfs: int
    points: float
    avgFinish: float | None = None


@strawberry.type
class Circuit:
    id: strawberry.ID
    name: str
    country: str | None = None
    lengthKm: float | None = None


@strawberry.type
class Event:
    id: strawberry.ID
    seasonId: strawberry.ID
    circuitId: strawberry.ID
    round: int
    name: str
    startDate: date | None = None


@strawberry.type
class Session:
    id: strawberry.ID
    eventId: strawberry.ID
    type: SessionTypeGQL
    dateStart: datetime | None = None
    openf1SessionKey: int | None = None


@strawberry.type
class Lap:
    id: strawberry.ID
    sessionId: strawberry.ID
    driverId: strawberry.ID
    lapNumber: int
    lapTimeMs: int | None = None
    sector1Ms: int | None = None
    sector2Ms: int | None = None
    sector3Ms: int | None = None
    compound: CompoundGQL
    tyreLife: int | None = None
    isPitOut: bool
    isPitIn: bool


@strawberry.type
class RaceResult:
    sessionId: strawberry.ID
    position: int | None
    grid: int | None
    points: float
    status: str | None
    fastestLapMs: int | None
    driver: Driver
    team: Team


@strawberry.type
class Season:
    id: strawberry.ID
    year: int


@strawberry.type
class StandingRow:
    position: int
    driver: Driver
    points: float
    wins: int
    podiums: int


# ---------------------------------------------------------------------------
# Analytics types
# ---------------------------------------------------------------------------


@strawberry.type
class PaceRow:
    driverId: strawberry.ID
    lap: int
    rollingSec: float


@strawberry.type
class GapRow:
    driverId: strawberry.ID
    lap: int
    gapSec: float


@strawberry.type
class UndercutEvent:
    lap: int
    type: str
    attackerDriverId: strawberry.ID
    victimDriverId: strawberry.ID


@strawberry.type
class DegradationRow:
    driverId: strawberry.ID
    stint: int
    compound: str
    lapsInStint: int
    degSecPerLap: float
    meanPaceSec: float | None
