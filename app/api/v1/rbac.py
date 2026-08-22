"""Admin-managed roles & permissions.

Read endpoints are open to any authenticated user (the dashboard needs the
matrix to render and to gate its own UI). Every mutation requires the
``roles:manage`` permission, is written to the audit trail with a before/after
diff, and busts the policy cache so the change is live immediately.

Guard rails that keep an admin from locking themselves out:

* system roles cannot be deleted, renamed or deactivated;
* the top-ranked role always keeps ``users:manage`` and ``roles:manage``;
* a role still assigned to users cannot be deleted;
* rank collisions are rejected (the hierarchy must stay a total order).
"""

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import delete, func, select

from app.api.deps import CurrentUser, DbSession, require_permission
from app.core.rbac_defaults import DEFAULT_GRANTS, DEFAULT_PERMISSIONS, DEFAULT_ROLES
from app.models import AuditLog, Permission, Profile, Role, RolePermission
from app.schemas.rbac import (
    GrantChange,
    MatrixOut,
    MatrixUpdate,
    MyAccessOut,
    PermissionCreate,
    PermissionOut,
    PermissionUpdate,
    RbacAuditOut,
    RoleCreate,
    RoleGrantsReplace,
    RoleOut,
    RoleUpdate,
)
from app.services import rbac

router = APIRouter(prefix="/rbac", tags=["rbac"])

CanManage = Annotated[Profile, Depends(require_permission("roles:manage"))]

RBAC_ACTION = "rbac.update"


def _ip(request: Request) -> str | None:
    return request.client.host if request.client else None


async def _audit(
    db: DbSession,
    request: Request,
    actor: Profile,
    *,
    entity: str,
    entity_id: str | None,
    detail: dict[str, Any],
) -> None:
    db.add(
        AuditLog(
            user_id=actor.id,
            action=RBAC_ACTION,
            entity=entity,
            entity_id=entity_id,
            detail=detail,
            ip_address=_ip(request),
        )
    )


async def _load_catalog(db: DbSession) -> tuple[list[Role], list[Permission], dict[str, set[str]]]:
    await _bootstrap_if_empty(db)
    roles = (
        (await db.execute(select(Role).order_by(Role.rank))).scalars().unique().all()
    )
    permissions = (
        (
            await db.execute(
                select(Permission).order_by(
                    Permission.group_label, Permission.sort_order, Permission.key
                )
            )
        )
        .scalars()
        .all()
    )
    perm_key = {p.id: p.key for p in permissions}
    links = (await db.execute(select(RolePermission))).scalars().all()
    role_name = {r.id: r.name for r in roles}
    matrix: dict[str, set[str]] = {r.name: set() for r in roles}
    for link in links:
        name, key = role_name.get(link.role_id), perm_key.get(link.permission_id)
        if name and key:
            matrix[name].add(key)
    return list(roles), list(permissions), matrix


async def _bootstrap_if_empty(db: DbSession) -> None:
    exists = (await db.execute(select(Role.id).limit(1))).first()
    if exists is None:
        await rbac.ensure_seeded(db)


async def _user_counts(db: DbSession) -> dict[str, int]:
    rows = (await db.execute(select(Profile.role, func.count()).group_by(Profile.role))).all()
    return {str(role): int(count) for role, count in rows}


# ── read ──────────────────────────────────────────────────────────────────


@router.get("/matrix", response_model=MatrixOut)
async def get_matrix(db: DbSession, user: CurrentUser) -> MatrixOut:
    """The full role × permission grid — the payload the admin console renders."""
    roles, permissions, matrix = await _load_catalog(db)
    counts = await _user_counts(db)
    policy = await rbac.get_policy(db, fresh=True)

    role_out = []
    for r in roles:
        item = RoleOut.model_validate(r)
        item.permissions = sorted(matrix.get(r.name, set()))
        item.user_count = counts.get(r.name, 0)
        item.locked_permissions = sorted(rbac.locked_permissions_for(policy, r.name))
        role_out.append(item)

    granted_by_key: dict[str, list[str]] = {p.key: [] for p in permissions}
    for name, keys in matrix.items():
        for key in keys:
            if key in granted_by_key:
                granted_by_key[key].append(name)

    perm_out = []
    for p in permissions:
        item = PermissionOut.model_validate(p)
        item.granted_to = sorted(granted_by_key.get(p.key, []))
        perm_out.append(item)

    groups: list[str] = []
    for p in permissions:
        if p.group_label not in groups:
            groups.append(p.group_label)

    latest = max((r.updated_at for r in roles), default=None)
    return MatrixOut(
        roles=role_out,
        permissions=perm_out,
        groups=groups,
        matrix={name: sorted(keys) for name, keys in matrix.items()},
        updated_at=latest,
    )


@router.get("/me", response_model=MyAccessOut)
@router.get("/rbac/me", response_model=MyAccessOut)
async def my_rbac_access(db: DbSession, user: CurrentUser) -> MyAccessOut:
    """Effective permissions of the caller — drives client-side UI gating."""
    await _bootstrap_if_empty(db)
    policy = await rbac.get_policy(db)
    info = policy.roles.get(user.role or "")
    return MyAccessOut(
        role=user.role,
        rank=policy.rank(user.role),
        label=info.label if info else None,
        permissions=sorted(policy.permissions_for(user.role)),
    )


@router.get("/roles", response_model=list[RoleOut])
async def list_roles(db: DbSession, user: CurrentUser) -> list[RoleOut]:
    matrix_out = await get_matrix(db, user)
    return matrix_out.roles


@router.get("/permissions", response_model=list[PermissionOut])
async def list_permissions(db: DbSession, user: CurrentUser) -> list[PermissionOut]:
    matrix_out = await get_matrix(db, user)
    return matrix_out.permissions


@router.get("/audit", response_model=list[RbacAuditOut])
async def rbac_audit(
    db: DbSession, user: CanManage, limit: int = Query(50, ge=1, le=200)
) -> list[RbacAuditOut]:
    """Change history for the matrix itself — who granted what, when."""
    logs = (
        (
            await db.execute(
                select(AuditLog)
                .where(AuditLog.action == RBAC_ACTION)
                .order_by(AuditLog.created_at.desc())
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    emails: dict[UUID, str] = {}
    out = []
    for log in logs:
        item = RbacAuditOut.model_validate(log)
        if log.user_id:
            if log.user_id not in emails:
                actor = await db.get(Profile, log.user_id)
                emails[log.user_id] = actor.email if actor else "(deleted user)"
            item.actor_email = emails[log.user_id]
        out.append(item)
    return out


# ── matrix editing ────────────────────────────────────────────────────────


async def _apply_changes(
    db: DbSession, actor: Profile, changes: list[GrantChange]
) -> list[dict[str, Any]]:
    roles = {r.name: r for r in (await db.execute(select(Role))).scalars().unique().all()}
    perms = {p.key: p for p in (await db.execute(select(Permission))).scalars().all()}
    policy = await rbac.get_policy(db, fresh=True)

    existing = {
        (link.role_id, link.permission_id): link
        for link in (await db.execute(select(RolePermission))).scalars().all()
    }

    applied: list[dict[str, Any]] = []
    for change in changes:
        role = roles.get(change.role)
        perm = perms.get(change.permission)
        if role is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown role '{change.role}'")
        if perm is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"Unknown permission '{change.permission}'"
            )
        if not change.granted and change.permission in rbac.locked_permissions_for(
            policy, role.name
        ):
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"'{change.permission}' cannot be revoked from '{role.name}' — "
                "removing it would leave nobody able to administer access.",
            )

        cell = (role.id, perm.id)
        currently = cell in existing
        if currently == change.granted:
            continue  # no-op, keep the audit diff meaningful
        if change.granted:
            link = RolePermission(role_id=role.id, permission_id=perm.id, granted_by=actor.id)
            db.add(link)
            existing[cell] = link
        else:
            await db.delete(existing.pop(cell))
        applied.append(
            {"role": role.name, "permission": perm.key, "granted": change.granted}
        )
    return applied


@router.patch("/matrix", response_model=MatrixOut)
async def update_matrix(
    body: MatrixUpdate, db: DbSession, user: CanManage, request: Request
) -> MatrixOut:
    """Apply a batch of grant/revoke toggles atomically."""
    applied = await _apply_changes(db, user, body.changes)
    if applied:
        await _audit(
            db,
            request,
            user,
            entity="matrix",
            entity_id=None,
            detail={"changes": applied, "count": len(applied)},
        )
    await db.commit()
    rbac.invalidate()
    return await get_matrix(db, user)


@router.put("/roles/{name}/permissions", response_model=RoleOut)
async def replace_role_permissions(
    name: str, body: RoleGrantsReplace, db: DbSession, user: CanManage, request: Request
) -> RoleOut:
    """Set a role's grants to exactly the supplied list."""
    role = (await db.execute(select(Role).where(Role.name == name))).scalar_one_or_none()
    if role is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Role not found")

    _, permissions, matrix = await _load_catalog(db)
    valid = {p.key for p in permissions}
    unknown = sorted(set(body.permissions) - valid)
    if unknown:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, f"Unknown permissions: {', '.join(unknown)}"
        )

    target = set(body.permissions)
    current = matrix.get(role.name, set())
    changes = [
        GrantChange(role=role.name, permission=key, granted=key in target)
        for key in valid
        if (key in target) != (key in current)
    ]
    applied = await _apply_changes(db, user, changes) if changes else []
    if applied:
        await _audit(
            db,
            request,
            user,
            entity="role",
            entity_id=role.name,
            detail={"replaced_grants": applied, "count": len(applied)},
        )
    await db.commit()
    rbac.invalidate()
    matrix_out = await get_matrix(db, user)
    return next(r for r in matrix_out.roles if r.name == role.name)


# ── role CRUD ─────────────────────────────────────────────────────────────


@router.post("/roles", response_model=RoleOut, status_code=status.HTTP_201_CREATED)
async def create_role(
    body: RoleCreate, db: DbSession, user: CanManage, request: Request
) -> RoleOut:
    await _bootstrap_if_empty(db)
    if (await db.execute(select(Role).where(Role.name == body.name))).scalar_one_or_none():
        raise HTTPException(status.HTTP_409_CONFLICT, f"Role '{body.name}' already exists")
    if (await db.execute(select(Role).where(Role.rank == body.rank))).scalar_one_or_none():
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Rank {body.rank} is taken — every role needs a distinct position in the hierarchy",
        )

    role = Role(
        name=body.name,
        label=body.label,
        description=body.description,
        rank=body.rank,
        color=body.color,
        is_system=False,
        is_active=body.is_active,
    )
    db.add(role)
    await db.flush()

    keys: list[str] = list(body.permissions or [])
    if body.clone_from:
        _, _, matrix = await _load_catalog(db)
        if body.clone_from not in matrix:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"Cannot clone unknown role '{body.clone_from}'"
            )
        keys = sorted(matrix[body.clone_from])

    if keys:
        perms = {
            p.key: p
            for p in (
                await db.execute(select(Permission).where(Permission.key.in_(keys)))
            ).scalars()
        }
        unknown = sorted(set(keys) - perms.keys())
        if unknown:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"Unknown permissions: {', '.join(unknown)}",
            )
        for key in keys:
            db.add(
                RolePermission(
                    role_id=role.id, permission_id=perms[key].id, granted_by=user.id
                )
            )

    await _audit(
        db,
        request,
        user,
        entity="role",
        entity_id=role.name,
        detail={
            "created": {"name": role.name, "label": role.label, "rank": role.rank},
            "permissions": keys,
            "cloned_from": body.clone_from,
        },
    )
    await db.commit()
    rbac.invalidate()
    matrix_out = await get_matrix(db, user)
    return next(r for r in matrix_out.roles if r.name == role.name)


@router.patch("/roles/{name}", response_model=RoleOut)
async def update_role(
    name: str, body: RoleUpdate, db: DbSession, user: CanManage, request: Request
) -> RoleOut:
    role = (await db.execute(select(Role).where(Role.name == name))).scalar_one_or_none()
    if role is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Role not found")

    changes = body.model_dump(exclude_unset=True)
    if role.is_system and changes.get("is_active") is False:
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"'{role.name}' is a system role and cannot be deactivated"
        )
    if "rank" in changes and changes["rank"] != role.rank:
        clash = (
            await db.execute(select(Role).where(Role.rank == changes["rank"], Role.id != role.id))
        ).scalar_one_or_none()
        if clash is not None:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"Rank {changes['rank']} is already used by '{clash.name}'",
            )

    before = {k: getattr(role, k) for k in changes}
    for field, value in changes.items():
        setattr(role, field, value)

    await _audit(
        db,
        request,
        user,
        entity="role",
        entity_id=role.name,
        detail={"before": _jsonable(before), "after": _jsonable(changes)},
    )
    await db.commit()
    rbac.invalidate()
    matrix_out = await get_matrix(db, user)
    return next(r for r in matrix_out.roles if r.name == role.name)


@router.delete("/roles/{name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_role(name: str, db: DbSession, user: CanManage, request: Request) -> None:
    role = (await db.execute(select(Role).where(Role.name == name))).scalar_one_or_none()
    if role is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Role not found")
    if role.is_system:
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"'{role.name}' is a system role and cannot be deleted"
        )
    assigned = (
        await db.execute(select(func.count()).select_from(Profile).where(Profile.role == name))
    ).scalar_one()
    if assigned:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"{assigned} user(s) still have the '{name}' role — reassign them first",
        )

    await _audit(
        db,
        request,
        user,
        entity="role",
        entity_id=role.name,
        detail={"deleted": {"name": role.name, "label": role.label, "rank": role.rank}},
    )
    await db.delete(role)
    await db.commit()
    rbac.invalidate()


# ── permission CRUD ───────────────────────────────────────────────────────


@router.post("/permissions", response_model=PermissionOut, status_code=status.HTTP_201_CREATED)
async def create_permission(
    body: PermissionCreate, db: DbSession, user: CanManage, request: Request
) -> PermissionOut:
    await _bootstrap_if_empty(db)
    clash = (
        await db.execute(select(Permission).where(Permission.key == body.key))
    ).scalar_one_or_none()
    if clash is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Permission '{body.key}' already exists")
    perm = Permission(
        key=body.key,
        label=body.label,
        description=body.description,
        group_label=body.group_label,
        sort_order=body.sort_order,
        is_system=False,
    )
    db.add(perm)
    await _audit(
        db,
        request,
        user,
        entity="permission",
        entity_id=body.key,
        detail={"created": body.model_dump()},
    )
    await db.commit()
    rbac.invalidate()
    await db.refresh(perm)
    return PermissionOut.model_validate(perm)


@router.patch("/permissions/{key:path}", response_model=PermissionOut)
async def update_permission(
    key: str, body: PermissionUpdate, db: DbSession, user: CanManage, request: Request
) -> PermissionOut:
    perm = (await db.execute(select(Permission).where(Permission.key == key))).scalar_one_or_none()
    if perm is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Permission not found")
    changes = body.model_dump(exclude_unset=True)
    before = {k: getattr(perm, k) for k in changes}
    for field, value in changes.items():
        setattr(perm, field, value)
    await _audit(
        db,
        request,
        user,
        entity="permission",
        entity_id=key,
        detail={"before": _jsonable(before), "after": _jsonable(changes)},
    )
    await db.commit()
    rbac.invalidate()
    await db.refresh(perm)
    return PermissionOut.model_validate(perm)


@router.delete("/permissions/{key:path}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_permission(
    key: str, db: DbSession, user: CanManage, request: Request
) -> None:
    perm = (await db.execute(select(Permission).where(Permission.key == key))).scalar_one_or_none()
    if perm is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Permission not found")
    if perm.is_system:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"'{key}' is enforced by the API and cannot be deleted — revoke it from every "
            "role instead.",
        )
    await db.execute(delete(RolePermission).where(RolePermission.permission_id == perm.id))
    await _audit(
        db, request, user, entity="permission", entity_id=key, detail={"deleted": {"key": key}}
    )
    await db.delete(perm)
    await db.commit()
    rbac.invalidate()


# ── reset ─────────────────────────────────────────────────────────────────


@router.post("/reset", response_model=MatrixOut)
async def reset_to_defaults(db: DbSession, user: CanManage, request: Request) -> MatrixOut:
    """Restore the shipped policy: re-grants defaults, drops custom roles' grants.

    Custom roles and custom permissions survive (they are the admin's own
    definitions); only the *grants* of the three system roles are rewritten to
    the values in ``app.core.rbac_defaults``.
    """
    _, permissions, matrix = await _load_catalog(db)
    perm_ids = {p.key: p.id for p in permissions}
    roles = {r.name: r for r in (await db.execute(select(Role))).scalars().unique().all()}

    reverted: list[dict[str, Any]] = []
    for seed in DEFAULT_ROLES:
        name = seed["name"]
        role = roles.get(name)
        if role is None:
            continue
        desired = {k for k in DEFAULT_GRANTS.get(name, []) if k in perm_ids}
        current = matrix.get(name, set())
        if desired == current:
            continue
        await db.execute(delete(RolePermission).where(RolePermission.role_id == role.id))
        for key in sorted(desired):
            db.add(
                RolePermission(
                    role_id=role.id, permission_id=perm_ids[key], granted_by=user.id
                )
            )
        reverted.append(
            {
                "role": name,
                "added": sorted(desired - current),
                "removed": sorted(current - desired),
            }
        )
        role.label = seed["label"]
        role.description = seed["description"]
        role.color = seed["color"]

    await _audit(
        db, request, user, entity="matrix", entity_id=None, detail={"reset_to_defaults": reverted}
    )
    await db.commit()
    rbac.invalidate()
    return await get_matrix(db, user)


@router.post("/sync-catalog", response_model=MatrixOut)
async def sync_catalog(db: DbSession, user: CanManage, request: Request) -> MatrixOut:
    """Insert any permission the code ships but the database lacks.

    Run after a deploy that introduced new capabilities: the new keys appear in
    the matrix ungranted, so an admin decides who gets them.
    """
    known = {
        k
        for (k,) in (await db.execute(select(Permission.key))).all()
    }
    missing = [p for p in DEFAULT_PERMISSIONS if p["key"] not in known]
    for seed in missing:
        db.add(
            Permission(
                key=seed["key"],
                label=seed["label"],
                description=seed["description"],
                group_label=seed["group_label"],
                sort_order=seed["sort_order"],
                is_system=True,
            )
        )
    if missing:
        await _audit(
            db,
            request,
            user,
            entity="matrix",
            entity_id=None,
            detail={"synced_permissions": [p["key"] for p in missing]},
        )
    await db.commit()
    rbac.invalidate()
    return await get_matrix(db, user)


def _jsonable(data: dict[str, Any]) -> dict[str, Any]:
    return {k: (str(v) if isinstance(v, UUID) else v) for k, v in data.items()}
