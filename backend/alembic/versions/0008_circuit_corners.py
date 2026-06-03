"""0008 circuit corners

Adds a ``corners`` JSONB column to ``circuits`` for the official FastF1 corner
list (turn number, letter, distance from start/finish in metres). Backfilled
by the ``seed-circuit-corners`` ETL command.

Schema of each element:
    {"number": 1, "letter": "", "distance_m": 350.0}

Revision ID: 0008
Revises: 0007
Create Date: 2026-06-10
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("circuits", sa.Column("corners", JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("circuits", "corners")
