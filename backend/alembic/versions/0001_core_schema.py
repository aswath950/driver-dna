"""0001 core schema

Creates the normalized v1 schema:
  ENUM types: session_type, compound_type
  Tables: seasons, circuits, events, sessions, teams, drivers,
          session_drivers, race_results, lap_times, driver_stats

Indexes are intentionally NOT added here — see migration 0002 (Phase 3).

Revision ID: 0001
Revises:
Create Date: 2026-05-24
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


SESSION_TYPE_VALUES = ("FP1", "FP2", "FP3", "Q", "SQ", "S", "R")
COMPOUND_TYPE_VALUES = ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN")


def upgrade() -> None:
    # ---- ENUM types -----------------------------------------------------
    session_type = postgresql.ENUM(*SESSION_TYPE_VALUES, name="session_type")
    compound_type = postgresql.ENUM(*COMPOUND_TYPE_VALUES, name="compound_type")
    session_type.create(op.get_bind(), checkfirst=True)
    compound_type.create(op.get_bind(), checkfirst=True)

    # ---- seasons --------------------------------------------------------
    op.create_table(
        "seasons",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("year", sa.Integer(), nullable=False),
        sa.UniqueConstraint("year", name="uq_seasons_year"),
    )

    # ---- circuits -------------------------------------------------------
    op.create_table(
        "circuits",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("country", sa.Text(), nullable=True),
        sa.Column("length_km", sa.Numeric(6, 3), nullable=True),
        sa.Column("sector_fractions", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )

    # ---- events ---------------------------------------------------------
    op.create_table(
        "events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "season_id",
            sa.Integer(),
            sa.ForeignKey("seasons.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "circuit_id",
            sa.Integer(),
            sa.ForeignKey("circuits.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("round", sa.Integer(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=True),
        sa.UniqueConstraint("season_id", "round", name="uq_events_season_round"),
    )

    # ---- sessions -------------------------------------------------------
    op.create_table(
        "sessions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "event_id",
            sa.Integer(),
            sa.ForeignKey("events.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "type",
            postgresql.ENUM(*SESSION_TYPE_VALUES, name="session_type", create_type=False),
            nullable=False,
        ),
        sa.Column("date_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("openf1_session_key", sa.BigInteger(), nullable=True),
        sa.UniqueConstraint("openf1_session_key", name="uq_sessions_openf1_key"),
    )

    # ---- teams ----------------------------------------------------------
    op.create_table(
        "teams",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("color_hex", sa.String(length=7), nullable=True),
        sa.UniqueConstraint("name", name="uq_teams_name"),
    )

    # ---- drivers --------------------------------------------------------
    op.create_table(
        "drivers",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("code", sa.String(length=3), nullable=False),
        sa.Column("full_name", sa.Text(), nullable=False),
        sa.Column("nationality", sa.String(length=2), nullable=True),
        sa.Column(
            "current_team_id",
            sa.Integer(),
            sa.ForeignKey("teams.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.UniqueConstraint("code", name="uq_drivers_code"),
    )

    # ---- session_drivers ------------------------------------------------
    op.create_table(
        "session_drivers",
        sa.Column(
            "session_id",
            sa.Integer(),
            sa.ForeignKey("sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "driver_id",
            sa.Integer(),
            sa.ForeignKey("drivers.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "team_id",
            sa.Integer(),
            sa.ForeignKey("teams.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("car_number", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("session_id", "driver_id", name="pk_session_drivers"),
    )

    # ---- race_results ---------------------------------------------------
    op.create_table(
        "race_results",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "session_id",
            sa.Integer(),
            sa.ForeignKey("sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "driver_id",
            sa.Integer(),
            sa.ForeignKey("drivers.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("position", sa.Integer(), nullable=True),
        sa.Column("grid", sa.Integer(), nullable=True),
        sa.Column("points", sa.Numeric(5, 2), nullable=False, server_default="0"),
        sa.Column("status", sa.Text(), nullable=True),
        sa.Column("fastest_lap_ms", sa.Integer(), nullable=True),
        sa.UniqueConstraint(
            "session_id", "driver_id", name="uq_race_results_session_driver"
        ),
    )

    # ---- lap_times ------------------------------------------------------
    op.create_table(
        "lap_times",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "session_id",
            sa.Integer(),
            sa.ForeignKey("sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "driver_id",
            sa.Integer(),
            sa.ForeignKey("drivers.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("lap_number", sa.Integer(), nullable=False),
        sa.Column("lap_time_ms", sa.Integer(), nullable=True),
        sa.Column("sector1_ms", sa.Integer(), nullable=True),
        sa.Column("sector2_ms", sa.Integer(), nullable=True),
        sa.Column("sector3_ms", sa.Integer(), nullable=True),
        sa.Column(
            "compound",
            postgresql.ENUM(*COMPOUND_TYPE_VALUES, name="compound_type", create_type=False),
            nullable=True,
        ),
        sa.Column("tyre_life", sa.Integer(), nullable=True),
        sa.Column("is_pit_out", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("is_pit_in", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.UniqueConstraint(
            "session_id",
            "driver_id",
            "lap_number",
            name="uq_lap_times_session_driver_lap",
        ),
        sa.CheckConstraint("lap_number > 0", name="ck_lap_times_lap_number_positive"),
    )

    # ---- driver_stats ---------------------------------------------------
    op.create_table(
        "driver_stats",
        sa.Column(
            "driver_id",
            sa.Integer(),
            sa.ForeignKey("drivers.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "season_id",
            sa.Integer(),
            sa.ForeignKey("seasons.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("wins", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("podiums", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("poles", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("dnfs", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("points", sa.Numeric(7, 2), nullable=False, server_default="0"),
        sa.Column("avg_finish", sa.Numeric(4, 2), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("driver_id", "season_id", name="pk_driver_stats"),
    )


def downgrade() -> None:
    # Drop in reverse FK order.
    op.drop_table("driver_stats")
    op.drop_table("lap_times")
    op.drop_table("race_results")
    op.drop_table("session_drivers")
    op.drop_table("drivers")
    op.drop_table("teams")
    op.drop_table("sessions")
    op.drop_table("events")
    op.drop_table("circuits")
    op.drop_table("seasons")

    bind = op.get_bind()
    postgresql.ENUM(name="compound_type").drop(bind, checkfirst=True)
    postgresql.ENUM(name="session_type").drop(bind, checkfirst=True)
