from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

SLUG = r"^[a-z][a-z0-9_-]{1,31}$"
PERMISSION_KEY = r"^[a-z][a-z0-9-]*:[a-z][a-z0-9-]*$"


class RoleOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    label: str
    description: str | None
    rank: int
    color: str
    is_system: bool
    is_active: bool
    created_at: datetime
    updated_at: datetime
    permissions: list[str] = []
    user_count: int = 0
    locked_permissions: list[str] = []


class RoleCreate(BaseModel):
    name: str = Field(pattern=SLUG)
    label: str = Field(min_length=1, max_length=64)
    description: str | None = Field(default=None, max_length=500)
    rank: int = Field(ge=1, le=999)
    color: str = Field(default="slate", max_length=24)
    is_active: bool = True
    permissions: list[str] | None = None
    # Copy the grant set from an existing role instead of listing keys.
    clone_from: str | None = Field(default=None, pattern=SLUG)


class RoleUpdate(BaseModel):
    label: str | None = Field(default=None, min_length=1, max_length=64)
    description: str | None = Field(default=None, max_length=500)
    rank: int | None = Field(default=None, ge=1, le=999)
    color: str | None = Field(default=None, max_length=24)
    is_active: bool | None = None


class PermissionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    key: str
    label: str
    description: str | None
    group_label: str
    sort_order: int
    is_system: bool
    granted_to: list[str] = []


class PermissionCreate(BaseModel):
    key: str = Field(pattern=PERMISSION_KEY)
    label: str = Field(min_length=1, max_length=80)
    description: str | None = Field(default=None, max_length=500)
    group_label: str = Field(default="General", min_length=1, max_length=64)
    sort_order: int = Field(default=0, ge=0, le=9999)


class PermissionUpdate(BaseModel):
    label: str | None = Field(default=None, min_length=1, max_length=80)
    description: str | None = Field(default=None, max_length=500)
    group_label: str | None = Field(default=None, min_length=1, max_length=64)
    sort_order: int | None = Field(default=None, ge=0, le=9999)


class GrantChange(BaseModel):
    """One cell of the matrix."""

    role: str = Field(pattern=SLUG)
    permission: str = Field(pattern=PERMISSION_KEY)
    granted: bool


class MatrixUpdate(BaseModel):
    """Batched matrix edit — applied atomically so the UI never half-saves."""

    changes: list[GrantChange] = Field(min_length=1, max_length=2000)


class RoleGrantsReplace(BaseModel):
    permissions: list[str]


class MatrixOut(BaseModel):
    roles: list[RoleOut]
    permissions: list[PermissionOut]
    groups: list[str]
    # role name -> granted permission keys; denormalised for the client grid
    matrix: dict[str, list[str]]
    updated_at: datetime | None = None


class MyAccessOut(BaseModel):
    role: str | None
    rank: int
    label: str | None
    permissions: list[str]


class RbacAuditOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    action: str
    entity: str | None
    entity_id: str | None
    detail: dict | None
    created_at: datetime
    actor_email: str | None = None
