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
    # The DB-level default stays. Deploys are not atomic: while the old image is
    # still serving, it INSERTs organizations without this column, and a NOT NULL
    # column with no default turns every business registration into a 500 until
    # the new build lands. ``is_legacy`` keeps its default for the same reason.
    op.add_column(
        "organizations",
        sa.Column("is_personal", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )


def downgrade():
    op.drop_column("organizations", "is_personal")
