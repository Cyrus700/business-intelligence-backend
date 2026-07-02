from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select

from app.api.deps import DbSession, require_role
from app.models import Profile
from app.schemas.identity import ProfileOut, UserCreate, UserUpdate
from app.services.supabase_admin import (
    SupabaseAdmin,
    SupabaseAdminError,
    get_supabase_admin,
)

router = APIRouter(prefix="/users", tags=["users"], dependencies=[Depends(require_role("admin"))])

AdminApi = Annotated[SupabaseAdmin, Depends(get_supabase_admin)]


@router.get("", response_model=list[ProfileOut])
async def list_users(db: DbSession) -> list[ProfileOut]:
    rows = (await db.execute(select(Profile).order_by(Profile.created_at))).scalars().all()
    return [ProfileOut.model_validate(r) for r in rows]


@router.post("", response_model=ProfileOut, status_code=status.HTTP_201_CREATED)
async def create_user(body: UserCreate, db: DbSession, admin_api: AdminApi) -> ProfileOut:
    existing = await db.execute(select(Profile).where(Profile.email == body.email))
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "A user with this email already exists")
    try:
        user_id = await admin_api.create_user(body.email, body.password, body.role, body.full_name)
    except SupabaseAdminError as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(e)) from e

    profile = Profile(
        id=user_id,
        email=body.email,
        full_name=body.full_name,
        role=body.role,
        department=body.department,
    )
    db.add(profile)
    await db.commit()
    await db.refresh(profile)
    return ProfileOut.model_validate(profile)


@router.patch("/{user_id}", response_model=ProfileOut)
async def update_user(
    user_id: UUID, body: UserUpdate, db: DbSession, admin_api: AdminApi
) -> ProfileOut:
    profile = await db.get(Profile, user_id)
    if profile is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")

    changes = body.model_dump(exclude_unset=True)
    role_changed = "role" in changes and changes["role"] != profile.role
    for field, value in changes.items():
        setattr(profile, field, value)

    if role_changed:
        # keep the JWT app_metadata claim in sync for RLS (Phase 6)
        try:
            await admin_api.set_role(user_id, profile.role)
        except SupabaseAdminError as e:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(e)) from e

    await db.commit()
    await db.refresh(profile)
    return ProfileOut.model_validate(profile)
