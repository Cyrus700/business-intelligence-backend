"""add_data_quality_framework

Revision ID: a1b2c3d4e5f6
Revises: 511e98ca2e4e
Create Date: 2026-08-19 11:34:43.642546
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: str | None = "511e98ca2e4e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "data_quality_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_date", sa.Date(), nullable=False),
        sa.Column("score", sa.Numeric(5, 2), nullable=False),
        sa.Column("dimensions", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("breakdown", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("rows_checked", sa.Integer(), nullable=False),
        sa.Column("issues_found", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("triggered_by", sa.String(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint("triggered_by IN ('schedule', 'manual', 'etl')", name="valid_trigger"),
    )
    op.create_index("ix_dq_runs_date", "data_quality_runs", ["run_date"])
    op.create_index(
        "ix_data_quality_runs_created_at",
        "data_quality_runs",
        ["created_at"],
    )

    op.create_table(
        "data_quality_issues",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("table_name", sa.String(), nullable=False),
        sa.Column("dimension", sa.String(), nullable=False),
        sa.Column("issue_type", sa.String(), nullable=False),
        sa.Column("severity", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("scope_key", sa.String(), nullable=True),
        sa.Column("scope_label", sa.String(), nullable=True),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("sample", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("resolved_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("resolved_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["data_quality_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["resolved_by"], ["profiles.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "dimension IN ('completeness', 'validity', 'consistency', 'uniqueness', 'timeliness', 'accuracy')",
            name="valid_dimension",
        ),
        sa.CheckConstraint(
            "issue_type IN ('null_required', 'negative_value', 'zero_quantity', "
            "'invalid_category', 'invalid_date', 'expected_total_mismatch', 'duplicate_row', "
            "'orphan_fk', 'stale_ingestion', 'threshold_degradation')",
            name="valid_issue_type",
        ),
        sa.CheckConstraint("severity IN ('info', 'warning', 'critical')", name="valid_severity"),
        sa.CheckConstraint("status IN ('open', 'acknowledged', 'resolved')", name="valid_status"),
    )
    op.create_index("ix_data_quality_issues_run_id", "data_quality_issues", ["run_id"])
    op.create_index("ix_dq_issues_status_severity", "data_quality_issues", ["status", "severity"])
    op.create_index("ix_data_quality_issues_created_at", "data_quality_issues", ["created_at"])


def downgrade() -> None:
    op.drop_table("data_quality_issues")
    op.drop_table("data_quality_runs")
