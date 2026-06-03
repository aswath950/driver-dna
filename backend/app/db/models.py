from __future__ import annotations

import enum
import uuid as _uuid
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    PrimaryKeyConstraint,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ENUM as PgEnum
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

# ---------------------------------------------------------------------------
# Phase 9 — llm_audit (one row per LLM call across all features).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ENUM types (created explicitly in migration 0001; ``create_type=False``
# here prevents SQLAlchemy from issuing CREATE TYPE during ``metadata.create_all``).
# ---------------------------------------------------------------------------


class SessionType(str, enum.Enum):
    FP1 = "FP1"
    FP2 = "FP2"
    FP3 = "FP3"
    Q = "Q"
    SQ = "SQ"
    S = "S"
    R = "R"


class CompoundType(str, enum.Enum):
    SOFT = "SOFT"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    INTERMEDIATE = "INTERMEDIATE"
    WET = "WET"
    UNKNOWN = "UNKNOWN"


session_type_pg = PgEnum(
    SessionType,
    name="session_type",
    create_type=False,
    values_callable=lambda x: [e.value for e in x],
)
compound_type_pg = PgEnum(
    CompoundType,
    name="compound_type",
    create_type=False,
    values_callable=lambda x: [e.value for e in x],
)


# ---------------------------------------------------------------------------
# Models (in dependency order to keep the file readable; Alembic ignores order)
# ---------------------------------------------------------------------------


class Season(Base):
    __tablename__ = "seasons"
    __table_args__ = (UniqueConstraint("year", name="uq_seasons_year"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    year: Mapped[int] = mapped_column(Integer, nullable=False)

    events: Mapped[list[Event]] = relationship(back_populates="season")


class Circuit(Base):
    __tablename__ = "circuits"
    __table_args__ = (UniqueConstraint("name", name="uq_circuits_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    country: Mapped[str | None] = mapped_column(Text)
    length_km: Mapped[Decimal | None] = mapped_column(Numeric(6, 3))
    sector_fractions: Mapped[list | None] = mapped_column(JSONB)
    x: Mapped[list | None] = mapped_column(JSONB)
    y: Mapped[list | None] = mapped_column(JSONB)
    corners: Mapped[list | None] = mapped_column(JSONB)

    events: Mapped[list[Event]] = relationship(back_populates="circuit")


class Event(Base):
    __tablename__ = "events"
    __table_args__ = (UniqueConstraint("season_id", "round", name="uq_events_season_round"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    season_id: Mapped[int] = mapped_column(
        ForeignKey("seasons.id", ondelete="CASCADE"), nullable=False
    )
    circuit_id: Mapped[int] = mapped_column(
        ForeignKey("circuits.id", ondelete="RESTRICT"), nullable=False
    )
    round: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    start_date: Mapped[date | None] = mapped_column(Date)

    season: Mapped[Season] = relationship(back_populates="events")
    circuit: Mapped[Circuit] = relationship(back_populates="events")
    sessions: Mapped[list[Session]] = relationship(back_populates="event")


class Session(Base):
    """Race weekend session (FP, Q, S, R). Named ``Session`` in line with the
    GraphQL schema; do not confuse with ``sqlalchemy.orm.Session``.
    """

    __tablename__ = "sessions"
    __table_args__ = (
        UniqueConstraint("openf1_session_key", name="uq_sessions_openf1_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[int] = mapped_column(
        ForeignKey("events.id", ondelete="CASCADE"), nullable=False
    )
    type: Mapped[SessionType] = mapped_column(session_type_pg, nullable=False)
    date_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    openf1_session_key: Mapped[int | None] = mapped_column(BigInteger)
    telemetry_fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    event: Mapped[Event] = relationship(back_populates="sessions")
    session_drivers: Mapped[list[SessionDriver]] = relationship(back_populates="session")
    results: Mapped[list[RaceResult]] = relationship(back_populates="session")
    laps: Mapped[list[LapTime]] = relationship(back_populates="session")


class Team(Base):
    __tablename__ = "teams"
    __table_args__ = (UniqueConstraint("name", name="uq_teams_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    color_hex: Mapped[str | None] = mapped_column(String(7))


class Driver(Base):
    __tablename__ = "drivers"
    __table_args__ = (UniqueConstraint("code", name="uq_drivers_code"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(3), nullable=False)
    full_name: Mapped[str] = mapped_column(Text, nullable=False)
    nationality: Mapped[str | None] = mapped_column(String(2))
    current_team_id: Mapped[int | None] = mapped_column(
        ForeignKey("teams.id", ondelete="SET NULL")
    )

    current_team: Mapped[Team | None] = relationship()


class SessionDriver(Base):
    """Association: which drivers participated in which session, for which team."""

    __tablename__ = "session_drivers"
    __table_args__ = (
        PrimaryKeyConstraint("session_id", "driver_id", name="pk_session_drivers"),
    )

    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    driver_id: Mapped[int] = mapped_column(
        ForeignKey("drivers.id", ondelete="RESTRICT"), nullable=False
    )
    team_id: Mapped[int] = mapped_column(
        ForeignKey("teams.id", ondelete="RESTRICT"), nullable=False
    )
    car_number: Mapped[int | None] = mapped_column(Integer)

    session: Mapped[Session] = relationship(back_populates="session_drivers")
    driver: Mapped[Driver] = relationship()
    team: Mapped[Team] = relationship()


class RaceResult(Base):
    __tablename__ = "race_results"
    __table_args__ = (
        UniqueConstraint("session_id", "driver_id", name="uq_race_results_session_driver"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    driver_id: Mapped[int] = mapped_column(
        ForeignKey("drivers.id", ondelete="RESTRICT"), nullable=False
    )
    position: Mapped[int | None] = mapped_column(Integer)
    grid: Mapped[int | None] = mapped_column(Integer)
    points: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=False, default=0)
    status: Mapped[str | None] = mapped_column(Text)
    fastest_lap_ms: Mapped[int | None] = mapped_column(Integer)

    session: Mapped[Session] = relationship(back_populates="results")
    driver: Mapped[Driver] = relationship()


class LapTime(Base):
    __tablename__ = "lap_times"
    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "driver_id",
            "lap_number",
            name="uq_lap_times_session_driver_lap",
        ),
        CheckConstraint("lap_number > 0", name="ck_lap_times_lap_number_positive"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    driver_id: Mapped[int] = mapped_column(
        ForeignKey("drivers.id", ondelete="RESTRICT"), nullable=False
    )
    lap_number: Mapped[int] = mapped_column(Integer, nullable=False)
    lap_time_ms: Mapped[int | None] = mapped_column(Integer)
    sector1_ms: Mapped[int | None] = mapped_column(Integer)
    sector2_ms: Mapped[int | None] = mapped_column(Integer)
    sector3_ms: Mapped[int | None] = mapped_column(Integer)
    compound: Mapped[CompoundType | None] = mapped_column(compound_type_pg)
    tyre_life: Mapped[int | None] = mapped_column(Integer)
    is_pit_out: Mapped[bool] = mapped_column(nullable=False, default=False)
    is_pit_in: Mapped[bool] = mapped_column(nullable=False, default=False)

    session: Mapped[Session] = relationship(back_populates="laps")
    driver: Mapped[Driver] = relationship()


class DriverStats(Base):
    """Materialised aggregate; refreshed by ETL job (Phase 5)."""

    __tablename__ = "driver_stats"
    __table_args__ = (
        PrimaryKeyConstraint("driver_id", "season_id", name="pk_driver_stats"),
    )

    driver_id: Mapped[int] = mapped_column(
        ForeignKey("drivers.id", ondelete="CASCADE"), nullable=False
    )
    season_id: Mapped[int] = mapped_column(
        ForeignKey("seasons.id", ondelete="CASCADE"), nullable=False
    )
    wins: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    podiums: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    poles: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    dnfs: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    points: Mapped[Decimal] = mapped_column(Numeric(7, 2), nullable=False, default=0)
    avg_finish: Mapped[Decimal | None] = mapped_column(Numeric(4, 2))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    driver: Mapped[Driver] = relationship()
    season: Mapped[Season] = relationship()


class LLMAudit(Base):
    __tablename__ = "llm_audit"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    request_id: Mapped[str | None] = mapped_column(Text)
    feature: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(Text, nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, server_default="0", nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, server_default="0", nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, server_default="0", nullable=False)
    cost_usd: Mapped[Decimal] = mapped_column(
        Numeric(10, 6), server_default="0", nullable=False
    )
    status: Mapped[str] = mapped_column(Text, nullable=False)
    error_type: Mapped[str | None] = mapped_column(Text)
    user_session_id: Mapped[_uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("user_sessions.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# ---------------------------------------------------------------------------
# Phase 10 — user_sessions + saved_analyses (anonymous personalisation).
# ---------------------------------------------------------------------------


class AnalysisKind(str, enum.Enum):
    RADAR = "radar"
    REPORT_CARD = "report_card"
    RACE_CHAT = "race_chat"
    XAI = "xai"
    DNA_MATCH = "dna_match"


class UserSession(Base):
    __tablename__ = "user_sessions"

    id: Mapped[_uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    ua: Mapped[str | None] = mapped_column(Text)
    locale: Mapped[str | None] = mapped_column(Text)


class SavedAnalysis(Base):
    __tablename__ = "saved_analyses"

    id: Mapped[_uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    user_session_id: Mapped[_uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("user_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    kind: Mapped[AnalysisKind] = mapped_column(
        PgEnum(AnalysisKind, name="analysis_kind", create_type=False, values_callable=lambda e: [m.value for m in e]),
        nullable=False,
    )
    session_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("sessions.id", ondelete="SET NULL"),
        nullable=True,
    )
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# ---------------------------------------------------------------------------
# Car telemetry cache — raw OpenF1 samples per session × driver × lap.
# ---------------------------------------------------------------------------


class CarTelemetry(Base):
    """Cached raw car_data samples from OpenF1, stored as columnar JSONB arrays.

    ``samples`` shape::

        {
          "dates":    ["2024-05-26T15:00:00.027Z", ...],
          "speed":    [152.3, ...],
          "throttle": [100, ...],
          "brake":    [0, ...],
          "rpm":      [11400, ...],
          "n_gear":   [6, ...],
          "drs":      [1, ...]
        }

    Reconstruct as a DataFrame with::

        pd.DataFrame({**samples, "date": pd.to_datetime(samples["dates"], utc=True)})
    """

    __tablename__ = "car_telemetry"
    __table_args__ = (
        PrimaryKeyConstraint("session_id", "driver_id", "lap_number",
                             name="pk_car_telemetry"),
    )

    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    driver_id: Mapped[int] = mapped_column(
        ForeignKey("drivers.id"), nullable=False
    )
    lap_number: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    lap_duration: Mapped[float | None] = mapped_column(Float)
    samples: Mapped[dict] = mapped_column(JSONB, nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
