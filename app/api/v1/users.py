from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

import bcrypt
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import func, or_, select

from app.api.deps import CurrentUser, DbSession, require_role
from app.models import AuditLog, Profile
from app.schemas.identity import ProfileOut, UserCreate, UserUpdate
from app.services import rbac

router = APIRouter(prefix="/users", tags=["users"], dependencies=[Depends(require_role("admin"))])


def _get_ip(request: Request) -> str | None:
    return request.client.host if request.client else None


async def _validate_role(db: DbSession, role: str) -> None:
    """Roles are admin-defined, so check the live catalog instead of a Literal."""
    policy = await rbac.get_policy(db, fresh=True)
    info = policy.roles.get(role)
    if info is None:
        known = ", ".join(sorted(policy.roles))
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"Unknown role '{role}'. Defined roles: {known}",
        )
    if not info.is_active:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Role '{role}' is deactivated and cannot be assigned")


async def _org_scoped_stmt(db: DbSession, user: Profile) -> Any:
    from app.api.deps import is_super_admin, org_predicate

    if is_super_admin(user):
        return None
    return org_predicate(Profile.org_id, user.org_id)


@router.get("")
async def list_users(
    db: DbSession,
    user: CurrentUser,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    search: str | None = None,
) -> dict:
    from app.api.deps import is_super_admin, org_predicate

    if is_super_admin(user):
        base = select(Profile)
    else:
        base = select(Profile).where(org_predicate(Profile.org_id, user.org_id))
    if search:
        q = f"%{search}%"
        base = base.where(or_(Profile.email.ilike(q), Profile.full_name.ilike(q)))
    count_q = select(func.count()).select_from(base.subquery())
    total = (await db.execute(count_q)).scalar_one()
    rows = (
        (await db.execute(base.order_by(Profile.created_at).offset((page - 1) * page_size).limit(page_size)))
        .scalars()
        .all()
    )
    return {
        "items": [ProfileOut.model_validate(r).model_dump() for r in rows],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/{user_id}", response_model=ProfileOut)
async def get_user(user_id: UUID, db: DbSession, user: CurrentUser) -> ProfileOut:
    from app.api.deps import is_super_admin, org_predicate

    if is_super_admin(user):
        profile = await db.get(Profile, user_id)
    else:
        scope = org_predicate(Profile.org_id, user.org_id)
        profile = (await db.execute(select(Profile).where(Profile.id == user_id, scope))).scalar_one_or_none()
    if profile is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    return ProfileOut.model_validate(profile)


@router.post("", response_model=ProfileOut, status_code=status.HTTP_201_CREATED)
async def create_user(body: UserCreate, db: DbSession, request: Request, user: CurrentUser) -> ProfileOut:
    await _validate_role(db, body.role)
    normalized_body_email = body.email.strip().lower()
    existing = await db.execute(select(Profile).where(func.lower(Profile.email) == normalized_body_email))
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "A user with this email already exists")

    # Password policy — same as signup / register-org
    if len(body.password) < 8:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Password must be at least 8 characters")
    if not (any(c.isalpha() for c in body.password) and any(c.isdigit() for c in body.password)):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "Password must contain at least one letter and one number"
        )

    from app.api.deps import is_super_admin

    # Enforce org scoping: never trust client-supplied org_id unless super_admin
    if is_super_admin(user) and body.org_id is not None:
        target_org = body.org_id
    else:
        target_org = user.org_id
        if body.org_id is not None and body.org_id != target_org:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Cannot create user in another organization")

    # Generate deterministic id from email (same as signup) so re-attempts collide cleanly
    user_id = uuid5(NAMESPACE_URL, f"email://{normalized_body_email}")
    # If somehow that id already exists (edge: different casing collision handled above),
    # fall back to random v4
    existing_id = await db.get(Profile, user_id)
    if existing_id is not None:
        import uuid

        user_id = uuid.uuid4()

    pw_hash = bcrypt.hashpw(body.password.encode(), bcrypt.gensalt(rounds=12)).decode()

    profile = Profile(
        id=user_id,
        email=normalized_body_email,
        password_hash=pw_hash,
        full_name=body.full_name,
        role=body.role,
        department=body.department,
        org_id=target_org,
        is_active=True,
        email_verified=True,
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
    user_id: UUID,
    body: UserUpdate,
    db: DbSession,
    request: Request,
    user: CurrentUser,
) -> ProfileOut:
    from app.api.deps import is_super_admin, org_predicate

    if is_super_admin(user):
        profile = await db.get(Profile, user_id)
    else:
        profile = (
            await db.execute(select(Profile).where(Profile.id == user_id, org_predicate(Profile.org_id, user.org_id)))
        ).scalar_one_or_none()
    if profile is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")

    changes = body.model_dump(exclude_unset=True)
    # Prevent self-deactivation — an admin locking themselves out is always an accident,
    # and with only one admin left it would brick the workspace.
    if changes.get("is_active") is False and user_id == user.id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "You cannot deactivate your own account")
    # Prevent org hopping via PATCH unless super_admin
    if "org_id" in changes and changes["org_id"] is not None:
        if is_super_admin(user):
            pass  # super_admin can move users
        elif changes["org_id"] != user.org_id:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Cannot move user to another organization")
    previous_role = profile.role
    role_changed = "role" in changes and changes["role"] != previous_role
    if role_changed:
        await _validate_role(db, changes["role"])
    for field, value in changes.items():
        setattr(profile, field, value)

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
