import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import ForeignKey, Index, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, uuid_pk

# Roles seeded on every deployment. The full catalog is admin-editable and
# lives in the ``roles`` table (app.models.rbac) — this tuple only names the
# ones the platform itself depends on.
ROLES = ("admin", "manager", "analyst")


class Organization(Base, TimestampMixin):
    """Tenant root. Null org_id on a table row means "default org" (single-tenant)."""

    __tablename__ = "organizations"

    id: Mapped[uuid.UUID] = uuid_pk()
    name: Mapped[str] = mapped_column(unique=True)


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
    # attribute-based access: tenant scope and token version for instant revocation
    org_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="SET NULL")
    )
    token_version: Mapped[int] = mapped_column(default=0)

    # No CHECK on `role`: custom roles are defined at runtime in the `roles`
    # table, so validity is enforced by the API against that catalog instead.
    __table_args__ = (Index("ix_profiles_org_id", "org_id"),)


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

    __table_args__ = (
        Index("ix_audit_logs_created_at", "created_at"),
        Index("ix_audit_logs_user_created", "user_id", "created_at"),
    )
