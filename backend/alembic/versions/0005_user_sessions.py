"""0005 user_sessions + saved_analyses (+ llm_audit FK).

Adds anonymous per-browser identity (``user_sessions``) and a list of
``saved_analyses`` rows scoped to that identity. Also backfills the
``llm_audit`` table with an optional FK so we can join calls to the
session that triggered them.

Revision ID: 0005
Revises: 0004
Create Date: 2026-05-26
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ENUM as PgEnum
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Postgres 13+ has gen_random_uuid() in pgcrypto.
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    op.create_table(
        "user_sessions",
        sa.Column(
            "id", UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            nullable=False, server_default=sa.text("now()"),
        ),
        sa.Column(
            "last_seen_at", sa.DateTime(timezone=True),
            nullable=False, server_default=sa.text("now()"),
        ),
        sa.Column("ua", sa.Text(), nullable=True),
        sa.Column("locale", sa.Text(), nullable=True),
    )

    kind_enum = PgEnum(
        "radar", "report_card", "race_chat", "xai", "dna_match",
        name="analysis_kind",
        create_type=True,
    )
    kind_enum.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "saved_analyses",
        sa.Column(
            "id", UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "user_session_id", UUID(as_uuid=True),
            sa.ForeignKey("user_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "kind",
            PgEnum(
                "radar", "report_card", "race_chat", "xai", "dna_match",
                name="analysis_kind",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column(
            "session_id", sa.Integer(),
            sa.ForeignKey("sessions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("payload", JSONB(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            nullable=False, server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_saved_analyses_user_session_created",
        "saved_analyses",
        ["user_session_id", sa.text("created_at DESC")],
    )

    # FK on llm_audit so we can attribute calls to a session.
    op.add_column(
        "llm_audit",
        sa.Column(
            "user_session_id", UUID(as_uuid=True),
            sa.ForeignKey("user_sessions.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_llm_audit_user_session_created",
        "llm_audit",
        ["user_session_id", sa.text("created_at DESC")],
    )


def downgrade() -> None:
    op.drop_index("ix_llm_audit_user_session_created", table_name="llm_audit")
    op.drop_column("llm_audit", "user_session_id")
    op.drop_index(
        "ix_saved_analyses_user_session_created", table_name="saved_analyses"
    )
    op.drop_table("saved_analyses")
    op.execute("DROP TYPE analysis_kind")
    op.drop_table("user_sessions")
