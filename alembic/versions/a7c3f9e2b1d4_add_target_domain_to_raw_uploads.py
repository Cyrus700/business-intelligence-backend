"""add target_domain to raw_uploads

Revision ID: a7c3f9e2b1d4
Revises: 75d6a4fce3bd
Create Date: 2026-08-05
"""

import sqlalchemy as sa

from alembic import op

revision = "a7c3f9e2b1d4"
down_revision = "75d6a4fce3bd"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "raw_uploads",
        sa.Column("target_domain", sa.String(20), nullable=True),
        schema="staging",
    )


def downgrade() -> None:
    op.drop_column("raw_uploads", "target_domain", schema="staging")
