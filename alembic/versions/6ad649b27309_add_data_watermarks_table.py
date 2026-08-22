"""add_data_watermarks_table

Revision ID: 6ad649b27309
Revises: b7a6c5d4e3f2
Create Date: 2026-08-19 15:10:37.468656
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


revision: str = '6ad649b27309'
down_revision: str | None = 'b7a6c5d4e3f2'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "data_watermarks",
        sa.Column("id", sa.Integer, primary_key=True, default=1),
        sa.Column(
            "last_refresh_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("last_source", sa.String(50), nullable=False),
        sa.Column("last_trigger", sa.String(50), nullable=False),
        sa.Column("affected_range_start", sa.Date, nullable=True),
        sa.Column("affected_range_end", sa.Date, nullable=True),
        sa.Column("details", JSONB, nullable=True),
        sa.CheckConstraint("id = 1", name="single_watermark_row"),
    )


def downgrade() -> None:
    op.drop_table("data_watermarks")
