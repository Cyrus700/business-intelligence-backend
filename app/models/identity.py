import uuid
from datetime import datetime
from typing import Any

from typing import Any

from sqlalchemy import CheckConstraint, ForeignKey, Index, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, uuid_pk

ROLES = ("admin", "manager", "analyst")


class Profile(Base, TimestampMixin):
    """1:1 extension of Supabase auth.users (same id)."""

    __tablename__ = "profiles"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    email: Mapped[str] = mapped_column(unique=True)
    password_hash: Mapped[str | None]
    full_name: Mapped[str | None]
    role: Mapped[str] = mapped_column(default="analyst")
    department: Mapped[str | None]
    is_active: Mapped[bool] = mapped_column(default=True)
    preferences: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)

    __table_args__ = (CheckConstraint(f"role IN {ROLES}", name="valid_role"),)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = uuid_pk()
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("profiles.id", ondelete="SET NULL")
    )
    action: Mapped[str]
    entity: Mapped[str | None]
    entity_id: Mapped[str | None]
    detail: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    ip_address: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (Index("ix_audit_logs_created_at", "created_at"),)
