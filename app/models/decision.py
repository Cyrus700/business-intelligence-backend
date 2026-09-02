import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import ARRAY, CheckConstraint, ForeignKey, Index, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, uuid_pk


class Insight(Base):
    __tablename__ = "insights"

    id: Mapped[uuid.UUID] = uuid_pk()
    insight_type: Mapped[str]
    title: Mapped[str]
    body: Mapped[str] = mapped_column(Text)
    severity: Mapped[str] = mapped_column(default="info")
    evidence: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    related_anomaly_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("anomalies.id", ondelete="SET NULL"))
    related_forecast_id: Mapped[int | None] = mapped_column(ForeignKey("forecasts.id", ondelete="SET NULL"))
    period_start: Mapped[date | None]
    period_end: Mapped[date | None]
    generated_at: Mapped[datetime] = mapped_column(server_default=func.now())
    is_pinned: Mapped[bool] = mapped_column(default=False)
    # Decision workflow (Phase 8): why -> impact -> priority -> decision.
    priority: Mapped[str] = mapped_column(default="medium")
    status: Mapped[str] = mapped_column(default="open")
    action: Mapped[str | None] = mapped_column(String(255))
    impact_estimate: Mapped[Decimal | None] = mapped_column(Numeric(18, 4))
    org_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE")
    )
    # dedupe key so re-runs don't duplicate insights (Phase 5)
    dedupe_key: Mapped[str | None] = mapped_column(unique=True)

    __table_args__ = (
        CheckConstraint(
            "insight_type IN ('trend', 'forecast', 'anomaly', 'comparison', 'recommendation')",
            name="valid_type",
        ),
        CheckConstraint("severity IN ('info', 'warning', 'critical')", name="valid_severity"),
        CheckConstraint(
            "status IN ('open', 'accepted', 'dismissed', 'postponed', 'actioned')",
            name="ck_insight_status",
        ),
        CheckConstraint("priority IN ('low', 'medium', 'high')", name="valid_priority"),
        Index("ix_insights_generated_at", "generated_at"),
        Index("ix_insights_type_generated", "insight_type", "generated_at"),
        Index("ix_insights_org_id", "org_id"),
        Index("ix_insights_org_type_generated", "org_id", "insight_type", "generated_at"),
    )


class AlertRule(Base, TimestampMixin):
    __tablename__ = "alert_rules"

    id: Mapped[uuid.UUID] = uuid_pk()
    name: Mapped[str]
    metric: Mapped[str]
    condition: Mapped[str]
    threshold: Mapped[Decimal | None] = mapped_column(Numeric(18, 4))
    window_days: Mapped[int] = mapped_column(default=7)
    channels: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    roles_notified: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="SET NULL")
    )
    org_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE")
    )

    __table_args__ = (
        CheckConstraint(
            "condition IN ('gt', 'lt', 'pct_change_gt', 'anomaly_detected')",
            name="valid_condition",
        ),
        Index("ix_alert_rules_org_id", "org_id"),
    )


class Notification(Base):
    __tablename__ = "notifications"

    id: Mapped[uuid.UUID] = uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="CASCADE"))
    alert_rule_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("alert_rules.id", ondelete="SET NULL"))
    insight_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("insights.id", ondelete="SET NULL"))
    title: Mapped[str]
    body: Mapped[str | None] = mapped_column(Text)
    is_read: Mapped[bool] = mapped_column(default=False)
    org_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE")
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (
        Index("ix_notifications_user_unread", "user_id", "is_read"),
        Index("ix_notifications_org_id", "org_id"),
    )


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[uuid.UUID] = uuid_pk()
    report_type: Mapped[str]
    period_start: Mapped[date]
    period_end: Mapped[date]
    format: Mapped[str] = mapped_column(default="pdf")
    s3_key: Mapped[str | None]
    generated_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="SET NULL")
    )
    org_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE")
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "report_type IN ('weekly_summary', 'monthly_summary', 'custom')",
            name="valid_report_type",
        ),
        CheckConstraint("format IN ('pdf', 'xlsx')", name="valid_format"),
        Index("ix_reports_org_id", "org_id"),
    )


class ReportSchedule(Base, TimestampMixin):
    """A user's standing subscription to a recurring generated report.

    ``next_run_at`` is the sole scheduling source of truth — the daily worker
    (app.workers.scheduler._run_due_report_schedules) queries for rows due and
    advances it after each run, rather than re-deriving "is this due today"
    from cron math on every tick.
    """

    __tablename__ = "report_schedules"

    id: Mapped[uuid.UUID] = uuid_pk()
    frequency: Mapped[str]
    format: Mapped[str] = mapped_column(default="pdf")
    day_of_week: Mapped[int | None]  # 0=Monday .. 6=Sunday, for weekly
    day_of_month: Mapped[int | None]  # 1..28, for monthly
    is_active: Mapped[bool] = mapped_column(default=True)
    next_run_at: Mapped[date]
    last_run_at: Mapped[datetime | None]
    last_report_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("reports.id", ondelete="SET NULL")
    )
    created_by: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="CASCADE"))
    org_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE")
    )

    __table_args__ = (
        CheckConstraint("frequency IN ('weekly', 'monthly')", name="valid_frequency"),
        CheckConstraint("format IN ('pdf', 'xlsx')", name="valid_schedule_format"),
        CheckConstraint("day_of_week IS NULL OR (day_of_week BETWEEN 0 AND 6)", name="valid_day_of_week"),
        CheckConstraint("day_of_month IS NULL OR (day_of_month BETWEEN 1 AND 28)", name="valid_day_of_month"),
        Index("ix_report_schedules_due", "is_active", "next_run_at"),
        Index("ix_report_schedules_org_id", "org_id"),
    )


class RecommendationFeedback(Base):
    """User outcome signal for recommendations (accepted / dismissed).

    Ranks future recommendations by prior acceptance (4a in the advance plan):
    keyed by the recommendation's dedupe_key (which encodes its type + scope),
    so feedback aggregates across re-runs.
    """

    __tablename__ = "recommendation_feedback"

    id: Mapped[uuid.UUID] = uuid_pk()
    rec_key: Mapped[str] = mapped_column(index=True)  # dedupe_key of the Insight
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="SET NULL")
    )
    org_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE")
    )
    action: Mapped[str]  # 'accepted' | 'dismissed' | 'postponed' | 'actioned'
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "action IN ('accepted', 'dismissed', 'postponed', 'actioned')",
            name="valid_action",
        ),
        CheckConstraint("LENGTH(rec_key) > 0", name="ck_rec_key_not_empty"),
        Index("ix_rec_feedback_key_created", "rec_key", "created_at"),
        Index("ix_rec_feedback_org_id", "org_id"),
    )
