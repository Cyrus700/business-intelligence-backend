"""Database-backed RBAC catalog.

Replaces the hard-coded ``ROLE_RANK`` / permission dictionaries with three
tables an admin can edit at runtime:

``roles``            the role catalog, ordered by ``rank`` (hierarchy preserved)
``permissions``      the capability catalog, grouped for presentation
``role_permissions`` the many-to-many grant matrix

System rows (``is_system``) are seeded from :mod:`app.core.rbac_defaults` and
are protected from deletion/rename so the auth layer always has the roles it
depends on.
"""

import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, ForeignKey, Index, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, uuid_pk


class Role(Base, TimestampMixin):
    __tablename__ = "roles"

    id: Mapped[uuid.UUID] = uuid_pk()
    name: Mapped[str] = mapped_column(unique=True)  # slug, matches profiles.role
    label: Mapped[str]
    description: Mapped[str | None]
    # Hierarchy: a higher rank implies "at least" every lower-ranked role for
    # require_role() checks. Unique so the ladder stays unambiguous.
    rank: Mapped[int] = mapped_column(unique=True)
    color: Mapped[str] = mapped_column(default="slate")
    is_system: Mapped[bool] = mapped_column(default=False)
    is_active: Mapped[bool] = mapped_column(default=True)

    grants: Mapped[list["RolePermission"]] = relationship(
        back_populates="role", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (
        CheckConstraint("rank > 0", name="positive_rank"),
        CheckConstraint("name ~ '^[a-z][a-z0-9_-]{1,31}$'", name="slug_name"),
    )


class Permission(Base, TimestampMixin):
    __tablename__ = "permissions"

    id: Mapped[uuid.UUID] = uuid_pk()
    key: Mapped[str] = mapped_column(unique=True)  # "resource:action"
    label: Mapped[str]
    description: Mapped[str | None]
    group_label: Mapped[str] = mapped_column(default="General")
    sort_order: Mapped[int] = mapped_column(default=0)
    is_system: Mapped[bool] = mapped_column(default=False)

    __table_args__ = (
        CheckConstraint("key ~ '^[a-z][a-z0-9-]*:[a-z][a-z0-9-]*$'", name="key_format"),
        Index("ix_permissions_group_label", "group_label"),
    )


class RolePermission(Base):
    __tablename__ = "role_permissions"

    id: Mapped[uuid.UUID] = uuid_pk()
    role_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("roles.id", ondelete="CASCADE"))
    permission_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("permissions.id", ondelete="CASCADE")
    )
    granted_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    role: Mapped[Role] = relationship(back_populates="grants")
    permission: Mapped[Permission] = relationship(lazy="selectin")

    __table_args__ = (
        UniqueConstraint("role_id", "permission_id", name="uq_role_permissions_role_id"),
        Index("ix_role_permissions_permission_id", "permission_id"),
    )
