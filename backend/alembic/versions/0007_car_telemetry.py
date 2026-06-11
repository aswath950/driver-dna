"""0007 car telemetry cache

Adds ``car_telemetry`` table to store raw OpenF1 car_data samples
(speed, throttle, brake, rpm, n_gear, drs) per session × driver × lap.
Also adds ``sessions.telemetry_fetched_at`` to track when a session's
telemetry was last downloaded.

Revision ID: 0007
Revises: 0006
Create Date: 2026-05-31
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "car_telemetry",
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("driver_id", sa.Integer(), nullable=False),
        sa.Column("lap_number", sa.SmallInteger(), nullable=False),
        sa.Column("lap_duration", sa.Float(), nullable=True),
        sa.Column("samples", JSONB(), nullable=False),
        sa.Column(
            "fetched_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["driver_id"], ["drivers.id"]),
        sa.PrimaryKeyConstraint("session_id", "driver_id", "lap_number",
                                name="pk_car_telemetry"),
    )
    op.create_index("idx_car_telemetry_session", "car_telemetry", ["session_id"])

    op.add_column(
        "sessions",
        sa.Column("telemetry_fetched_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("sessions", "telemetry_fetched_at")
    op.drop_index("idx_car_telemetry_session", table_name="car_telemetry")
    op.drop_table("car_telemetry")
