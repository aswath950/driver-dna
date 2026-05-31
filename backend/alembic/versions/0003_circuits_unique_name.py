"""0003 circuits unique name

Adds the UNIQUE constraint on ``circuits.name`` that the ETL needs as an
ON CONFLICT target. Omitted from 0001 because Phase 2 spec didn't list it.

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-25
"""

from __future__ import annotations

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_unique_constraint("uq_circuits_name", "circuits", ["name"])


def downgrade() -> None:
    op.drop_constraint("uq_circuits_name", "circuits", type_="unique")
