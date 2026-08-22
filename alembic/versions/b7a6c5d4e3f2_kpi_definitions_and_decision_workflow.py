"""kpi definitions, model lifecycle, insight decision workflow, indexes

Revision ID: b7a6c5d4e3f2
Revises: f9c8e7d6a5b4
Create Date: 2026-08-19

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "b7a6c5d4e3f2"
down_revision: str | None = "f9c8e7d6a5b4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "kpi_definitions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("metric", sa.String(64), nullable=False),
        sa.Column("label", sa.String(120), nullable=False),
        sa.Column("formula", sa.String(255), nullable=False),
        sa.Column("unit", sa.String(16), nullable=False, server_default=""),
        sa.Column("higher_is_better", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("target_value", sa.Numeric(18, 4), nullable=True),
        sa.Column("threshold_low", sa.Numeric(18, 4), nullable=True),
        sa.Column("owner_id", UUID(as_uuid=True), nullable=True),
        sa.Column("visibility", sa.ARRAY(sa.String()), nullable=False, server_default=sa.text("ARRAY['analyst']")),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("metric", name="uq_kpi_definition_metric"),
    )
    op.create_check_constraint(
        "ck_kpi_metric_not_empty", "kpi_definitions", "char_length(metric) > 0"
    )

    op.add_column("ml_models", sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("ml_models", sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True))

    op.add_column("insights", sa.Column("priority", sa.String(8), nullable=False, server_default="medium"))
    op.add_column("insights", sa.Column("status", sa.String(12), nullable=False, server_default="open"))
    op.add_column("insights", sa.Column("action", sa.String(255), nullable=True))
    op.add_column("insights", sa.Column("impact_estimate", sa.Numeric(18, 4), nullable=True))
    op.create_check_constraint(
        "ck_insight_status",
        "insights",
        "status IN ('open', 'accepted', 'dismissed', 'postponed', 'actioned')",
    )
    op.create_check_constraint(
        "valid_priority", "insights", "priority IN ('low', 'medium', 'high')"
    )

    op.drop_constraint("valid_action", "recommendation_feedback", type_="check")
    op.create_check_constraint(
        "valid_action",
        "recommendation_feedback",
        "action IN ('accepted', 'dismissed', 'postponed', 'actioned')",
    )

    # Performance indexes on hot read paths (Phase 14).
    op.create_index(
        "ix_kpi_snapshots_metric_dims", "kpi_snapshots", ["metric", "dimensions"]
    )
    op.create_index("ix_forecasts_model_id", "forecasts", ["model_id"])
    op.create_index("ix_insights_type_generated", "insights", ["insight_type", "generated_at"])
    op.create_index("ix_etl_jobs_status_started", "etl_jobs", ["status", "started_at"])
    op.create_index("ix_audit_logs_user_created", "audit_logs", ["user_id", "created_at"])

    # Seed metadata-driven KPI definitions for the metrics kpi_builder computes.
    kpi_defs = sa.table(
        "kpi_definitions",
        sa.column("id", sa.Uuid()),
        sa.column("metric", sa.String()),
        sa.column("label", sa.String()),
        sa.column("formula", sa.String()),
        sa.column("unit", sa.String()),
        sa.column("higher_is_better", sa.Boolean()),
        sa.column("visibility", sa.ARRAY(sa.String())),
    )
    import uuid as _uuid

    op.bulk_insert(
        kpi_defs,
        [
            {
                "id": _uuid.uuid4(),
                "metric": "revenue",
                "label": "Revenue",
                "formula": "SUM(sales_transactions.total_amount)",
                "unit": "NPR",
                "higher_is_better": True,
                "visibility": ["analyst"],
            },
            {
                "id": _uuid.uuid4(),
                "metric": "orders",
                "label": "Orders",
                "formula": "COUNT(sales_transactions.id)",
                "unit": "orders",
                "higher_is_better": True,
                "visibility": ["analyst"],
            },
            {
                "id": _uuid.uuid4(),
                "metric": "avg_order_value",
                "label": "Avg order value",
                "formula": "AVG(sales_transactions.total_amount)",
                "unit": "NPR",
                "higher_is_better": True,
                "visibility": ["analyst"],
            },
            {
                "id": _uuid.uuid4(),
                "metric": "gross_margin",
                "label": "Gross margin",
                "formula": "SUM(total_amount - COALESCE(unit_cost, 0) * quantity)",
                "unit": "NPR",
                "higher_is_better": True,
                "visibility": ["manager"],
            },
            {
                "id": _uuid.uuid4(),
                "metric": "expense_total",
                "label": "Total expenses",
                "formula": "SUM(expenses.amount)",
                "unit": "NPR",
                "higher_is_better": False,
                "visibility": ["analyst"],
            },
            {
                "id": _uuid.uuid4(),
                "metric": "stockout_count",
                "label": "Stockouts",
                "formula": "COUNT(*) FILTER (WHERE quantity_on_hand <= reorder_level)",
                "unit": "SKUs",
                "higher_is_better": False,
                "visibility": ["analyst"],
            },
        ],
    )


def downgrade() -> None:
    op.drop_index("ix_audit_logs_user_created", table_name="audit_logs")
    op.drop_index("ix_etl_jobs_status_started", table_name="etl_jobs")
    op.drop_index("ix_insights_type_generated", table_name="insights")
    op.drop_index("ix_forecasts_model_id", table_name="forecasts")
    op.drop_index("ix_kpi_snapshots_metric_dims", table_name="kpi_snapshots")

    op.drop_constraint("valid_action", "recommendation_feedback", type_="check")
    op.create_check_constraint(
        "valid_action", "recommendation_feedback", "action IN ('accepted', 'dismissed')"
    )
    op.drop_constraint("valid_priority", "insights", type_="check")
    op.drop_constraint("ck_insight_status", "insights", type_="check")
    op.drop_column("insights", "impact_estimate")
    op.drop_column("insights", "action")
    op.drop_column("insights", "status")
    op.drop_column("insights", "priority")
    op.drop_column("ml_models", "retired_at")
    op.drop_column("ml_models", "activated_at")
    op.drop_table("kpi_definitions")