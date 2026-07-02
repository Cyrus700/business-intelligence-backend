from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr

Role = Literal["admin", "manager", "analyst"]


class ProfileOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: str
    full_name: str | None
    role: Role
    department: str | None
    is_active: bool
    created_at: datetime


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str | None = None
    role: Role = "analyst"
    department: str | None = None


class UserUpdate(BaseModel):
    full_name: str | None = None
    role: Role | None = None
    department: str | None = None
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
