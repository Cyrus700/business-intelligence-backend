from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import func, or_, select
from sqlalchemy.orm import load_only

from app.api.deps import CurrentUser, DbSession, require_role
from app.models import AuditLog, Profile
from app.schemas.identity import ProfileOut, UserCreate, UserUpdate
from app.services import rbac
from app.services.supabase_admin import (
    SupabaseAdmin,
    SupabaseAdminError,
    get_supabase_admin,
)

router = APIRouter(prefix="/users", tags=["users"], dependencies=[Depends(require_role("admin"))])

AdminApi = Annotated[SupabaseAdmin, Depends(get_supabase_admin)]


def _get_ip(request: Request) -> str | None:
    return request.client.host if request.client else None


async def _validate_role(db: DbSession, role: str) -> None:
    """Roles are admin-defined, so check the live catalog instead of a Literal."""
    policy = await rbac.get_policy(db, fresh=True)
    info = policy.roles.get(role)
    if info is None:
        known = ", ".join(sorted(policy.roles))
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Unknown role '{role}'. Defined roles: {known}",
        )
    if not info.is_active:
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"Role '{role}' is deactivated and cannot be assigned"
        )


async def _org_scoped_stmt(db: DbSession, user: Profile) -> Any:
    from app.api.deps import org_predicate, user_org_id

    return org_predicate(Profile.org_id, user_org_id(user))


@router.get("")
async def list_users(
    db: DbSession,
    user: CurrentUser,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    search: str | None = None,
) -> dict:
    from app.api.deps import org_predicate, user_org_id

    base = select(Profile).where(org_predicate(Profile.org_id, user_org_id(user)))
    if search:
        q = f"%{search}%"
        base = base.where(or_(Profile.email.ilike(q), Profile.full_name.ilike(q)))
    count_q = select(func.count()).select_from(base.subquery())
    total = (await db.execute(count_q)).scalar_one()
    rows = (
        await db.execute(
            base.order_by(Profile.created_at)
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
    ).scalars().all()
    return {
        "items": [ProfileOut.model_validate(r).model_dump() for r in rows],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/{user_id}", response_model=ProfileOut)
async def get_user(user_id: UUID, db: DbSession, user: CurrentUser) -> ProfileOut:
    from app.api.deps import org_predicate, user_org_id

    scope = org_predicate(Profile.org_id, user_org_id(user))
    profile = (
        await db.execute(select(Profile).where(Profile.id == user_id, scope))
    ).scalar_one_or_none()
    if profile is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    return ProfileOut.model_validate(profile)


@router.post("", response_model=ProfileOut, status_code=status.HTTP_201_CREATED)
async def create_user(
    body: UserCreate, db: DbSession, admin_api: AdminApi, request: Request, user: CurrentUser
) -> ProfileOut:
    await _validate_role(db, body.role)
    existing = await db.execute(select(Profile).where(Profile.email == body.email))
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "A user with this email already exists")
    try:
        user_id = await admin_api.create_user(body.email, body.password, body.role, body.full_name)
    except SupabaseAdminError as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(e)) from e

    from app.api.deps import user_org_id

    profile = Profile(
        id=user_id,
        email=body.email,
        full_name=body.full_name,
        role=body.role,
        department=body.department,
        org_id=body.org_id or user_org_id(user),
    )
    db.add(profile)
    await db.flush()
    db.add(
        AuditLog(
            user_id=request.state.user.id if hasattr(request.state, "user") else None,
            action="POST /api/v1/users",
            entity="user",
            entity_id=str(user_id),
            detail={"email": body.email, "role": body.role, "org_id": str(profile.org_id or "")},
            ip_address=_get_ip(request),
        )
    )
    await db.commit()
    await db.refresh(profile)
    return ProfileOut.model_validate(profile)


@router.patch("/{user_id}", response_model=ProfileOut)
async def update_user(
    user_id: UUID, body: UserUpdate, db: DbSession, admin_api: AdminApi, request: Request,
    user: CurrentUser,
) -> ProfileOut:
    from app.api.deps import org_predicate, user_org_id

    profile = (
        await db.execute(
            select(Profile).where(
                Profile.id == user_id, org_predicate(Profile.org_id, user_org_id(user))
            )
        )
    ).scalar_one_or_none()
    if profile is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")

    changes = body.model_dump(exclude_unset=True)
    previous_role = profile.role
    role_changed = "role" in changes and changes["role"] != previous_role
    if role_changed:
        await _validate_role(db, changes["role"])
    for field, value in changes.items():
        setattr(profile, field, value)

    if role_changed:
        try:
            await admin_api.set_role(user_id, profile.role)
        except SupabaseAdminError as e:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(e)) from e

    # session revocation: role change or account disable invalidates every
    # outstanding JWT (the "ver" claim in security.py vs token_version here)
    revokes_session = role_changed or (changes.get("is_active") is False)
    if revokes_session:
        profile.token_version += 1

    log_detail = dict(changes)
    if role_changed:
        log_detail["previous_role"] = previous_role
    if revokes_session:
        log_detail["session_revoked"] = True
        log_detail["token_version"] = profile.token_version
    log_detail["by_role"] = user.role

    db.add(
        AuditLog(
            user_id=request.state.user.id if hasattr(request.state, "user") else None,
            action=f"PATCH /api/v1/users/{user_id}",
            entity="user",
            entity_id=str(user_id),
            detail=log_detail,
            ip_address=_get_ip(request),
        )
    )
    await db.commit()
    await db.refresh(profile)
    return ProfileOut.model_validate(profile)
