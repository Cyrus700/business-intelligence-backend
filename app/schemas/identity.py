from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field

# Roles are admin-defined at runtime (see app.models.rbac), so the wire type is
# a slug and the API validates it against the `roles` catalog rather than a
# closed Literal.
Role = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{1,31}$")]


class ProfileOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: str
    full_name: str | None
    role: Role
    department: str | None
    org_id: UUID | None = None
    is_super_admin: bool = False
    is_active: bool
    email_verified: bool = False
    created_at: datetime


class OrganizationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    slug: str | None = None
    is_legacy: bool = False
    status: str = "pending"
    approved_at: datetime | None = None
    approved_by: UUID | None = None
    rejected_at: datetime | None = None
    rejected_by: UUID | None = None
    rejection_reason: str | None = None
    created_at: datetime


class OrganizationInviteOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    org_id: UUID
    email: str | None
    role: Role
    token: str
    created_by: UUID | None
    expires_at: datetime
    accepted_at: datetime | None
    created_at: datetime


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str | None = None
    role: Role = "analyst"
    department: str | None = None
    org_id: UUID | None = None  # tenant; defaults to the creating admin's org


class UserUpdate(BaseModel):
    full_name: str | None = None
    role: Role | None = None
    department: str | None = None
    org_id: UUID | None = None
    is_active: bool | None = None


class AuditLogOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID | None
    action: str
    entity: str | None
    entity_id: str | None
    detail: dict[str, Any] | None
    ip_address: str | None
    created_at: datetime
