"""add_ai_retention_settings

Revision ID: b4c8d9e0f1a2
Revises: f9c8e7d6a5b4
Create Date: 2026-09-02
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "b4c8d9e0f1a2"
down_revision: str | None = "f9c8e7d6a5b4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ai_retention_settings",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("retention_days", sa.Integer(), nullable=False, server_default=sa.text("30")),
        sa.Column("updated_by", sa.UUID(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "retention_days >= 0 AND retention_days <= 365",
            name=op.f("ck_ai_retention_settings_ck_retention_valid_days"),
        ),
        sa.ForeignKeyConstraint(
            ["updated_by"],
            ["profiles.id"],
            name=op.f("fk_ai_retention_settings_updated_by_profiles"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_ai_retention_settings")),
    )


def downgrade() -> None:
    op.drop_table("ai_retention_settings")
