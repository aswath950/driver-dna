"""0004 llm_audit table

Persists one row per LLM call across all 5 agentic features so we can track
token cost, latency, and per-feature reliability over time.

Revision ID: 0004
Revises: 0003
Create Date: 2026-05-26
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "llm_audit",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("request_id", sa.Text(), nullable=True),
        sa.Column("feature", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("latency_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cost_usd", sa.Numeric(10, 6), nullable=False, server_default="0"),
        sa.Column("status", sa.Text(), nullable=False),  # 'success' | 'error'
        sa.Column("error_type", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_llm_audit_feature_created",
        "llm_audit",
        ["feature", sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_llm_audit_created",
        "llm_audit",
        [sa.text("created_at DESC")],
    )


def downgrade() -> None:
    op.drop_index("ix_llm_audit_created", table_name="llm_audit")
    op.drop_index("ix_llm_audit_feature_created", table_name="llm_audit")
    op.drop_table("llm_audit")
