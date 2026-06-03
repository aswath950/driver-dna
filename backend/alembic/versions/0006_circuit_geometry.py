"""0006 circuit geometry

Adds ``x`` and ``y`` JSONB columns to ``circuits`` for the track outline used
by the fastest-lap track-map chart. Backfilled by the ``seed-circuits`` ETL
command from ``data/circuits.json``.

Revision ID: 0006
Revises: 0005
Create Date: 2026-05-31
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("circuits", sa.Column("x", JSONB(), nullable=True))
    op.add_column("circuits", sa.Column("y", JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("circuits", "y")
    op.drop_column("circuits", "x")
