"""personal workspaces: self-serve accounts that are their own tenant

Revision ID: a3f5c81d6e07
Revises: f1a2b3c4d5e6
Create Date: 2026-09-02
"""

import sqlalchemy as sa

from alembic import op

revision = "a3f5c81d6e07"
down_revision = "f1a2b3c4d5e6"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "organizations",
        sa.Column("is_personal", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.alter_column("organizations", "is_personal", server_default=None)


def downgrade():
    op.drop_column("organizations", "is_personal")
