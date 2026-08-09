from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession, require_role
from app.models import AuditLog, Profile
from app.schemas.identity import AuditLogOut

router = APIRouter(
    prefix="/audit-logs", tags=["audit"], dependencies=[Depends(require_role("admin"))]
)


@router.get("", response_model=list[AuditLogOut])
async def list_audit_logs(
    db: DbSession,
    action: str | None = None,
    since: datetime | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
) -> list[AuditLogOut]:
    stmt = select(AuditLog).order_by(AuditLog.created_at.desc())
    if action:
        stmt = stmt.where(AuditLog.action == action)
    if since:
        stmt = stmt.where(AuditLog.created_at >= since)
    stmt = stmt.offset((page - 1) * page_size).limit(page_size)
    rows = (await db.execute(stmt)).scalars().all()
    return [AuditLogOut.model_validate(r) for r in rows]


class RoleChangeOut(AuditLogOut):
    actor_email: str | None = None
    subject_email: str | None = None


@router.get("/role-changes", response_model=list[RoleChangeOut])
async def list_role_changes(
    db: DbSession, user: CurrentUser, limit: int = Query(100, ge=1, le=500)
) -> list[RoleChangeOut]:
    """Who changed whose role, when — the RBAC audit trail for /permissions.

    Covers the admin user-management flow (PATCH /users/{id}) and any audit
    row whose detail mentions a role change. Org-scoped so multi-tenant admins
    only see their own org's trail.
    """
    from app.api.deps import org_predicate, user_org_id

    scope = org_predicate(Profile.org_id, user_org_id(user))
    logs = (
        (
            await db.execute(
                select(AuditLog)
                .where(
                    AuditLog.action.like("PATCH /api/v1/users/%"),
                    AuditLog.user_id.in_(
                        select(Profile.id).where(scope)
                    ),
                )
                .order_by(AuditLog.created_at.desc())
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    out = []
    for log in logs:
        detail = log.detail or {}
        if "role" not in detail and "previous_role" not in detail:
            continue
        item = RoleChangeOut.model_validate(log)
        actor = log.user_id
        if actor:
            actor_row = await db.get(Profile, actor)
            item.actor_email = actor_row.email if actor_row else None
        item.subject_email = None
        submitted = await db.get(Profile, log.entity_id) if log.entity_id else None
        if submitted:
            item.subject_email = submitted.email
        out.append(item)
    return out
