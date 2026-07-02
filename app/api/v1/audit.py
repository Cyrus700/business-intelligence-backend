from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select

from app.api.deps import DbSession, require_role
from app.models import AuditLog
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
