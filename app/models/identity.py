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
    """Tenant root. Every tenant owns one row; all tenant-scoped tables carry org_id FK."""

    __tablename__ = "organizations"

    id: Mapped[uuid.UUID] = uuid_pk()
    name: Mapped[str] = mapped_column(unique=True)
    # Optional slug/code for human-friendly invites (e.g. "acme-2026")
    slug: Mapped[str | None] = mapped_column(unique=True)
    is_legacy: Mapped[bool] = mapped_column(default=False)
    # Approval workflow (system admin must approve new businesses)
    status: Mapped[str] = mapped_column(default="pending")  # pending | approved | rejected
    approved_at: Mapped[datetime | None]
    approved_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="SET NULL")
    )
    rejected_at: Mapped[datetime | None]
    rejected_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="SET NULL")
    )
    rejection_reason: Mapped[str | None]


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
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="RESTRICT")
    )
    # Platform operator can see/manage all orgs; not tied to any single org's data
    is_super_admin: Mapped[bool] = mapped_column(default=False)
    token_version: Mapped[int] = mapped_column(default=0)
    # Email verification (business admin must verify email before approval)
    email_verified: Mapped[bool] = mapped_column(default=False)
    email_verification_token: Mapped[str | None] = mapped_column(unique=True)
    email_verification_expires_at: Mapped[datetime | None]
    email_verified_at: Mapped[datetime | None]

    # No CHECK on `role`: custom roles are defined at runtime in the `roles`
    # table, so validity is enforced by the API against that catalog instead.
    __table_args__ = (Index("ix_profiles_org_id", "org_id"),)


class OrganizationInvite(Base):
    """Invite token for a user to join an existing org (manager/analyst onboarding)."""

    __tablename__ = "organization_invites"

    id: Mapped[uuid.UUID] = uuid_pk()
    org_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"))
    email: Mapped[str | None]
    role: Mapped[str] = mapped_column(default="analyst")
    token: Mapped[str] = mapped_column(unique=True)  # opaque hex, delivered via API / email
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="SET NULL")
    )
    expires_at: Mapped[datetime] = mapped_column()
    accepted_at: Mapped[datetime | None]
    accepted_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("ix_org_invites_org_id", "org_id"),
        Index("ix_org_invites_token", "token"),
        Index("ix_org_invites_email", "email"),
    )


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = uuid_pk()
    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("profiles.id", ondelete="SET NULL"))
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
