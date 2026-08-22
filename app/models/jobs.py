import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import CheckConstraint, Index, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, uuid_pk

# In-process fallback dispatch (single instance) vs SQS-backed dispatch
# (horizontally scalable): dispatch_job() writes rows here when SQS is not
# configured, and ANY worker instance can claim them via advisory locks —
# durable across restarts, works on every instance, no extra infra.
JOB_STATUSES = ("pending", "claimed", "succeeded", "failed")


class BackgroundJob(Base, TimestampMixin):
    __tablename__ = "background_jobs"

    id: Mapped[uuid.UUID] = uuid_pk()
    name: Mapped[str]  # registered handler, e.g. "ml_retrain", "anomaly_scan"
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=dict)
    status: Mapped[str] = mapped_column(default="pending")
    run_at: Mapped[datetime] = mapped_column(server_default=func.now(), index=True)
    attempts: Mapped[int] = mapped_column(default=0)
    last_error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None]
    finished_at: Mapped[datetime | None]

    __table_args__ = (
        CheckConstraint("status IN ('pending', 'succeeded', 'failed')", name="valid_status"),
        Index("ix_background_jobs_pending", "status", "run_at"),
    )