"""add anomaly explanation and resolution

Revision ID: f9c8e7d6a5b4
Revises: a1b2c3d4e5f6
Create Date: 2026-08-19

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision: str = "f9c8e7d6a5b4"
down_revision: str | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "anomalies",
        sa.Column(
            "explanation",
            JSONB(),
            nullable=True,
            comment=(
                "Contributor analysis: which regions/channels/products/categories "
                "drove the deviation vs the trailing baseline window"
            ),
        ),
    )
    op.add_column("anomalies", sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column(
        "anomalies",
        sa.Column(
            "resolved_by",
            UUID(as_uuid=True),
            sa.ForeignKey("profiles.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.drop_constraint("valid_status", "anomalies", type_="check")
    op.create_check_constraint(
        "valid_status",
        "anomalies",
        "status IN ('open', 'acknowledged', 'dismissed', 'resolved')",
    )


def downgrade() -> None:
    op.drop_constraint("valid_status", "anomalies", type_="check")
    op.create_check_constraint("valid_status", "anomalies", "status IN ('open', 'acknowledged', 'dismissed')")
    op.drop_column("anomalies", "resolved_by")
    op.drop_column("anomalies", "resolved_at")
    op.drop_column("anomalies", "explanation")
