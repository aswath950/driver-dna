"""0002 indexes

Adds query-supporting indexes. Each index is justified by a real query in
``backend/docs/query_plans.md``.

Note: ``events(season_id, round)`` is intentionally NOT created here — the
``uq_events_season_round`` unique constraint from migration 0001 already
backs that lookup with a unique btree index. Adding a second composite
index would be pure waste.

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-24
"""

from __future__ import annotations

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Cross-driver lap views: "show me lap 23 for everyone in this session".
    op.create_index(
        "ix_lap_times_session_lap",
        "lap_times",
        ["session_id", "lap_number"],
    )

    # Pit-window queries (small selective subset of lap rows). Partial index
    # keeps the index tiny relative to the table.
    op.create_index(
        "ix_lap_times_pit",
        "lap_times",
        ["session_id"],
        postgresql_where="is_pit_in OR is_pit_out",
    )

    # Leaderboard: ORDER BY position for a given session.
    op.create_index(
        "ix_race_results_session_pos",
        "race_results",
        ["session_id", "position"],
    )

    # "Recent races" feed. DESC matters — Postgres can scan backward on a
    # plain ASC index, but a DESC index avoids the reversal cost on every
    # request.
    op.create_index(
        "ix_events_start_date",
        "events",
        [op.f("start_date")],
        postgresql_ops={"start_date": "DESC"},
    )

    # Standings ordering for a season.
    op.create_index(
        "ix_driver_stats_season_points",
        "driver_stats",
        ["season_id", "points"],
        postgresql_ops={"points": "DESC"},
    )


def downgrade() -> None:
    op.drop_index("ix_driver_stats_season_points", table_name="driver_stats")
    op.drop_index("ix_events_start_date", table_name="events")
    op.drop_index("ix_race_results_session_pos", table_name="race_results")
    op.drop_index("ix_lap_times_pit", table_name="lap_times")
    op.drop_index("ix_lap_times_session_lap", table_name="lap_times")
