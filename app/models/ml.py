import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import CheckConstraint, ForeignKey, Index, Numeric, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, uuid_pk

Metric = Numeric(18, 4)


class MlModel(Base):
    __tablename__ = "ml_models"

    id: Mapped[uuid.UUID] = uuid_pk()
    model_type: Mapped[str]
    target: Mapped[str]
    version: Mapped[int] = mapped_column(default=1)
    trained_at: Mapped[datetime] = mapped_column(server_default=func.now())
    training_rows: Mapped[int | None]
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    params: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    artifact_s3_key: Mapped[str | None]
    is_active: Mapped[bool] = mapped_column(default=False)

    __table_args__ = (
        CheckConstraint(
            "model_type IN ('prophet', 'arima', 'isolation_forest', 'regression_ensemble')",
            name="valid_model_type",
        ),
        UniqueConstraint("model_type", "target", "version", name="uq_model_version"),
    )


class Forecast(Base):
    __tablename__ = "forecasts"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("ml_models.id", ondelete="CASCADE"))
    target: Mapped[str]
    forecast_date: Mapped[date]
    horizon_days: Mapped[int]
    yhat: Mapped[Decimal] = mapped_column(Metric)
    yhat_lower: Mapped[Decimal | None] = mapped_column(Metric)
    yhat_upper: Mapped[Decimal | None] = mapped_column(Metric)
    generated_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (
        UniqueConstraint("model_id", "target", "forecast_date", name="uq_forecast_point"),
        Index("ix_forecasts_target_date", "target", "forecast_date"),
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
    status: Mapped[str] = mapped_column(default="open")
    acknowledged_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="SET NULL")
    )

    __table_args__ = (
        CheckConstraint("severity IN ('low', 'medium', 'high')", name="valid_severity"),
        CheckConstraint("status IN ('open', 'acknowledged', 'dismissed')", name="valid_status"),
        Index("ix_anomalies_status", "status"),
    )
