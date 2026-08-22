import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import CheckConstraint, ForeignKey, Index, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, uuid_pk

SOURCE_KINDS = ("csv_upload", "excel_upload", "rest_api", "postgres")
TARGET_DOMAINS = ("sales", "finance", "inventory")


class DataSource(Base, TimestampMixin):
    __tablename__ = "data_sources"

    id: Mapped[uuid.UUID] = uuid_pk()
    name: Mapped[str] = mapped_column(unique=True)
    kind: Mapped[str]
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    target_domain: Mapped[str]
    schedule_cron: Mapped[str | None]
    status: Mapped[str] = mapped_column(default="active")
    org_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("organizations.id"))

    __table_args__ = (
        CheckConstraint(f"kind IN {SOURCE_KINDS}", name="valid_kind"),
        CheckConstraint(f"target_domain IN {TARGET_DOMAINS}", name="valid_domain"),
        CheckConstraint("status IN ('active', 'paused', 'error')", name="valid_status"),
    )


class RawUpload(Base):
    __tablename__ = "raw_uploads"
    __table_args__ = (
        CheckConstraint(
            "status IN ('received', 'validated', 'loaded', 'failed')", name="valid_status"
        ),
        {"schema": "staging"},
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    data_source_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("data_sources.id", ondelete="SET NULL")
    )
    file_name: Mapped[str]
    target_domain: Mapped[str | None] = mapped_column(nullable=True)
    s3_key: Mapped[str | None]
    uploaded_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="SET NULL")
    )
    org_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("organizations.id"))
    row_count: Mapped[int | None]
    status: Mapped[str] = mapped_column(default="received")
    error_report: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class EtlJob(Base):
    __tablename__ = "etl_jobs"

    id: Mapped[uuid.UUID] = uuid_pk()
    data_source_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("data_sources.id", ondelete="SET NULL")
    )
    org_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("organizations.id"))
    trigger: Mapped[str]
    started_at: Mapped[datetime] = mapped_column(server_default=func.now())
    finished_at: Mapped[datetime | None]
    status: Mapped[str] = mapped_column(default="running")
    rows_in: Mapped[int | None]
    rows_loaded: Mapped[int | None]
    rows_rejected: Mapped[int | None]
    log: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    __table_args__ = (
        CheckConstraint("trigger IN ('manual', 'schedule', 'upload')", name="valid_trigger"),
        CheckConstraint("status IN ('running', 'succeeded', 'failed')", name="valid_status"),
        Index("ix_etl_jobs_status_started", "status", "started_at"),
    )
