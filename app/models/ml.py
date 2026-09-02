import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Numeric, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, uuid_pk

Metric = Numeric(18, 4)


class MlModel(Base):
    __tablename__ = "ml_models"

    id: Mapped[uuid.UUID] = uuid_pk()
    model_type: Mapped[str]
    target: Mapped[str]
    # segment filter, e.g. {"region": "Kathmandu"} or {} for whole-business
    # (mirrors kpi_snapshots.dimensions — same segment vocabulary throughout).
    dimensions: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    version: Mapped[int] = mapped_column(default=1)
    trained_at: Mapped[datetime] = mapped_column(server_default=func.now())
    training_rows: Mapped[int | None]
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    params: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    artifact_s3_key: Mapped[str | None]
    is_active: Mapped[bool] = mapped_column(default=False)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    org_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE")
    )

    __table_args__ = (
        CheckConstraint(
            "model_type IN ('prophet', 'arima', 'isolation_forest', 'regression_ensemble')",
            name="valid_model_type",
        ),
        UniqueConstraint("model_type", "target", "dimensions", "version", "org_id", name="uq_model_version"),
        Index("ix_ml_models_org_id", "org_id"),
        Index("ix_ml_models_org_target", "org_id", "target"),
    )


class Forecast(Base):
    __tablename__ = "forecasts"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("ml_models.id", ondelete="CASCADE"))
    target: Mapped[str]
    dimensions: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    forecast_date: Mapped[date]
    horizon_days: Mapped[int]
    yhat: Mapped[Decimal] = mapped_column(Metric)
    yhat_lower: Mapped[Decimal | None] = mapped_column(Metric)
    yhat_upper: Mapped[Decimal | None] = mapped_column(Metric)
    generated_at: Mapped[datetime] = mapped_column(server_default=func.now())
    org_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE")
    )

    __table_args__ = (
        UniqueConstraint(
            "model_id", "target", "dimensions", "forecast_date", name="uq_forecast_point"
        ),
        Index("ix_forecasts_target_date", "target", "forecast_date"),
        Index("ix_forecasts_model_id", "model_id"),
        Index("ix_forecasts_org_id", "org_id"),
        Index("ix_forecasts_org_target_date", "org_id", "target", "forecast_date"),
    )


class Anomaly(Base):
    __tablename__ = "anomalies"

    id: Mapped[uuid.UUID] = uuid_pk()
    detected_at: Mapped[datetime] = mapped_column(server_default=func.now())
    metric: Mapped[str]
    observed_value: Mapped[Decimal] = mapped_column(Metric)
    expected_value: Mapped[Decimal | None] = mapped_column(Metric)
    deviation_score: Mapped[Decimal | None] = mapped_column(Metric)
    severity: Mapped[str] = mapped_column(default="medium")
    context: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    explanation: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(default="open")
    acknowledged_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="SET NULL")
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolved_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="SET NULL")
    )
    org_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE")
    )
    # anomalies on the same business day across metrics are grouped into one
    # correlated event (root-cause-aware alerting)
    correlation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), index=True
    )

    __table_args__ = (
        CheckConstraint("severity IN ('low', 'medium', 'high')", name="valid_severity"),
        CheckConstraint(
            "status IN ('open', 'acknowledged', 'dismissed', 'resolved')",
            name="valid_status",
        ),
        Index("ix_anomalies_status", "status"),
        Index("ix_anomalies_org_id", "org_id"),
        Index("ix_anomalies_org_status", "org_id", "status"),
    )


class AnomalyFeedback(Base):
    """User verdict on a flagged anomaly (false positive vs confirmed).

    Feeds per-target detector calibration (services/ml/calibration.py): every
    dismissal/confirmation is persisted so the CONTAMINATION/Z_THRESHOLD tuning
    is explainable and reversible, consistent with the audit pattern.
    """

    __tablename__ = "anomaly_feedback"

    id: Mapped[uuid.UUID] = uuid_pk()
    anomaly_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("anomalies.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="SET NULL")
    )
    org_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE")
    )
    disposition: Mapped[str]  # 'false_positive' | 'confirmed'
    note: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "disposition IN ('false_positive', 'confirmed')", name="valid_disposition"
        ),
        Index("ix_anomaly_feedback_anomaly_created", "anomaly_id", "created_at"),
        Index("ix_anomaly_feedback_org_id", "org_id"),
    )


class ModelDrift(Base):
    """Live-vs-holdout forecast accuracy checkpoint.

    Every drift evaluation inserts a row (auditable history of why a model was
    retrained early); ``triggered`` marks the check that caused a retrain.
    """

    __tablename__ = "model_drift"

    id: Mapped[uuid.UUID] = uuid_pk()
    model_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("ml_models.id", ondelete="CASCADE"), index=True
    )
    target: Mapped[str]
    measured_on: Mapped[date] = mapped_column(index=True)
    window_days: Mapped[int] = mapped_column(default=7)
    live_mape: Mapped[Decimal | None] = mapped_column(Metric)
    holdout_mape: Mapped[Decimal | None] = mapped_column(Metric)
    threshold_mape: Mapped[Decimal | None] = mapped_column(Metric)
    triggered: Mapped[bool] = mapped_column(default=False)
    org_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE")
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (Index("ix_model_drift_org_id", "org_id"),)
