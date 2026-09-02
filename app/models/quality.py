import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import CheckConstraint, ForeignKey, Index, Numeric, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, uuid_pk

# Data Quality Framework (Phase 3 upgrade):
# DQ_RUN_DIMENSIONS — the six measurable quality dimensions. Each run scores
# 0..100 per dimension and a weighted overall score.
DQ_DIMENSIONS = (
    "completeness",
    "validity",
    "consistency",
    "uniqueness",
    "timeliness",
    "accuracy",
)

DQ_WEIGHTS = {
    "completeness": 0.25,
    "validity": 0.25,
    "uniqueness": 0.15,
    "consistency": 0.15,
    "timeliness": 0.10,
    "accuracy": 0.10,
}

DQ_ISSUE_TYPES = (
    "null_required",
    "negative_value",
    "zero_quantity",
    "invalid_category",
    "invalid_date",
    "expected_total_mismatch",
    "duplicate_row",
    "orphan_fk",
    "stale_ingestion",
    "threshold_degradation",
)

DQ_ISSUE_SEVERITIES = ("info", "warning", "critical")
DQ_ISSUE_STATUSES = ("open", "acknowledged", "resolved")


class DataQualityRun(Base):
    """One execution of the data-quality audit over the warehouse.

    Stores the overall score, per-dimension scores and a JSON breakdown
    (by table, by domain, by source) so the UI can show score + history
    without recomputing.
    """

    __tablename__ = "data_quality_runs"

    id: Mapped[uuid.UUID] = uuid_pk()
    run_date: Mapped[date] = mapped_column(index=True)
    score: Mapped[Decimal] = mapped_column(Numeric(5, 2))  # 0..100 overall
    dimensions: Mapped[dict[str, Any]] = mapped_column(JSONB)  # per-dimension scores
    breakdown: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)  # by table/domain/source
    rows_checked: Mapped[int] = mapped_column(default=0)
    issues_found: Mapped[int] = mapped_column(default=0)
    status: Mapped[str] = mapped_column(default="succeeded")  # succeeded | failed
    triggered_by: Mapped[str] = mapped_column(default="schedule")  # schedule | manual | etl
    duration_ms: Mapped[int] = mapped_column(default=0)
    error: Mapped[str | None] = mapped_column(Text)
    org_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE")
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (
        CheckConstraint("triggered_by IN ('schedule', 'manual', 'etl')", name="valid_trigger"),
        Index("ix_dq_runs_date", "run_date"),
        Index("ix_dq_runs_org_id", "org_id"),
    )


class DataQualityIssue(Base):
    """A specific, actionable quality problem discovered by an audit run.

    Issues are the analyst/admin-facing surface of the framework: each row
    names the table + dimension + issue type, how many rows were affected, a
    human-readable scope description and a status lifecycle (open →
    acknowledged → resolved).
    """

    __tablename__ = "data_quality_issues"

    id: Mapped[uuid.UUID] = uuid_pk()
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("data_quality_runs.id", ondelete="CASCADE"), index=True)
    table_name: Mapped[str]
    dimension: Mapped[str]
    issue_type: Mapped[str]
    severity: Mapped[str] = mapped_column(default="warning")
    status: Mapped[str] = mapped_column(default="open")
    scope_key: Mapped[str | None]  # e.g. "domain:sales", "source:<uuid>", "table:products"
    scope_label: Mapped[str | None]  # human-readable scope, e.g. "Sales domain"
    description: Mapped[str] = mapped_column(Text)
    row_count: Mapped[int] = mapped_column(default=0)
    sample: Mapped[dict[str, Any] | None] = mapped_column(JSONB)  # small sample of bad rows
    resolved_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="SET NULL")
    )
    resolved_at: Mapped[datetime | None]
    org_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE")
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (
        CheckConstraint(f"dimension IN {DQ_DIMENSIONS}", name="valid_dimension"),
        CheckConstraint(f"issue_type IN {DQ_ISSUE_TYPES}", name="valid_issue_type"),
        CheckConstraint(f"severity IN {DQ_ISSUE_SEVERITIES}", name="valid_severity"),
        CheckConstraint(f"status IN {DQ_ISSUE_STATUSES}", name="valid_status"),
        Index("ix_dq_issues_status_severity", "status", "severity"),
        Index("ix_dq_issues_org_id", "org_id"),
    )
