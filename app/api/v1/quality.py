"""Data Quality endpoints: score, history, issues, manual runs (Phase 3 upgrade).

Routes:
* GET  /data-quality/overview    — latest run: score, dimensions, breakdown
* GET  /data-quality/history     — recent audit runs
* GET  /data-quality/issues      — paginated issue list with filters
* POST /data-quality/run         — trigger a manual audit (manager+)
* PATCH /data-quality/issues/{id}— acknowledge / resolve an issue (analyst+)
"""

from datetime import date
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select

from app.api.deps import CurrentUser, DbSession, get_current_user, require_permission
from app.models import Profile
from app.models.quality import DataQualityIssue, DataQualityRun
from app.services.quality.engine import acknowledge_issue, run_quality_audit

CanRunQuality = Annotated[Profile, Depends(require_permission("quality:run"))]
CanResolveQuality = Annotated[Profile, Depends(require_permission("quality:resolve"))]

router = APIRouter(
    prefix="/data-quality",
    tags=["data-quality"],
    dependencies=[Depends(get_current_user)],
)


class DimensionScores(BaseModel):
    completeness: float | None = None
    validity: float | None = None
    consistency: float | None = None
    uniqueness: float | None = None
    timeliness: float | None = None
    accuracy: float | None = None


class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    run_date: date
    score: float
    dimensions: dict[str, Any]
    breakdown: dict[str, Any]
    rows_checked: int
    issues_found: int
    triggered_by: str
    duration_ms: int
    status: str
    created_at: Any


class IssueOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    run_id: UUID
    table_name: str
    dimension: str
    issue_type: str
    severity: str
    status: str
    scope_key: str | None
    scope_label: str | None
    description: str
    row_count: int
    sample: dict[str, Any] | None
    created_at: Any


class IssueUpdate(BaseModel):
    status: str  # acknowledged | resolved


class OverviewOut(BaseModel):
    latest: RunOut | None
    trend: list[RunOut]
    open_issues: int
    by_severity: dict[str, int]
    by_status: dict[str, int]
    by_dimension: dict[str, int]


@router.get("/overview", response_model=OverviewOut)
async def data_quality_overview(
    db: DbSession,
    user: CurrentUser,
    limit: int = Query(30, ge=1, le=90),
) -> OverviewOut:
    from app.api.deps import is_super_admin

    org_filter_run = [] if is_super_admin(user) else [DataQualityRun.org_id == user.org_id]
    org_filter_issue = [] if is_super_admin(user) else [DataQualityIssue.org_id == user.org_id]
    latest = (
        (await db.execute(select(DataQualityRun).where(*org_filter_run).order_by(DataQualityRun.created_at.desc())))
        .scalars()
        .first()
    )

    history = (
        (
            await db.execute(
                select(DataQualityRun).where(*org_filter_run).order_by(DataQualityRun.created_at.desc()).limit(limit)
            )
        )
        .scalars()
        .all()
    )

    severity_counts = dict(
        (
            await db.execute(
                select(DataQualityIssue.severity, func.count())
                .where(DataQualityIssue.status != "resolved", *org_filter_issue)
                .group_by(DataQualityIssue.severity)
            )
        ).all()
    )
    status_counts = dict(
        (
            await db.execute(
                select(DataQualityIssue.status, func.count()).where(*org_filter_issue).group_by(DataQualityIssue.status)
            )
        ).all()
    )
    dimension_counts = dict(
        (
            await db.execute(
                select(DataQualityIssue.dimension, func.count())
                .where(DataQualityIssue.status != "resolved", *org_filter_issue)
                .group_by(DataQualityIssue.dimension)
            )
        ).all()
    )
    open_issues = status_counts.get("open", 0) + status_counts.get("acknowledged", 0)

    return OverviewOut(
        latest=latest,
        trend=history,
        open_issues=int(open_issues),
        by_severity={k: int(v) for k, v in severity_counts.items()},
        by_status={k: int(v) for k, v in status_counts.items()},
        by_dimension={k: int(v) for k, v in dimension_counts.items()},
    )


@router.get("/quality/history", response_model=list[RunOut])
async def data_quality_history(
    db: DbSession,
    user: CurrentUser,
    limit: int = Query(60, ge=1, le=200),
) -> list[DataQualityRun]:
    from app.api.deps import is_super_admin

    q = select(DataQualityRun).order_by(DataQualityRun.created_at.desc()).limit(limit)
    if not is_super_admin(user):
        q = q.where(DataQualityRun.org_id == user.org_id)
    return list((await db.execute(q)).scalars())


@router.get("/issues", response_model=dict)
async def data_quality_issues(
    db: DbSession,
    user: CurrentUser,
    dimension: str | None = None,
    severity: str | None = None,
    status_filter: str | None = Query(None, alias="status"),
    table: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> dict[str, Any]:
    from app.api.deps import is_super_admin

    org_clause = [] if is_super_admin(user) else [DataQualityIssue.org_id == user.org_id]
    stmt = (
        select(DataQualityIssue)
        .where(*org_clause)
        .order_by(DataQualityIssue.created_at.desc(), DataQualityIssue.row_count.desc())
    )
    count_stmt = select(func.count()).select_from(DataQualityIssue).where(*org_clause)

    if dimension:
        stmt = stmt.where(DataQualityIssue.dimension == dimension)
        count_stmt = count_stmt.where(DataQualityIssue.dimension == dimension)
    if severity:
        stmt = stmt.where(DataQualityIssue.severity == severity)
        count_stmt = count_stmt.where(DataQualityIssue.severity == severity)
    if status_filter:
        stmt = stmt.where(DataQualityIssue.status == status_filter)
        count_stmt = count_stmt.where(DataQualityIssue.status == status_filter)
    if table:
        stmt = stmt.where(DataQualityIssue.table_name == table)
        count_stmt = count_stmt.where(DataQualityIssue.table_name == table)

    total = (await db.execute(count_stmt)).scalar_one()
    rows = (await db.execute(stmt.offset((page - 1) * page_size).limit(page_size))).scalars().all()

    return {
        "items": [IssueOut.model_validate(r).model_dump() for r in rows],
        "total": int(total),
        "page": page,
        "page_size": page_size,
    }


@router.post("/run", response_model=RunOut, status_code=status.HTTP_201_CREATED)
async def trigger_quality_audit(
    db: DbSession,
    user: CanRunQuality,
) -> DataQualityRun:
    run = await run_quality_audit(db, triggered_by="manual", org_id=user.org_id)
    if run is None:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Quality audit failed")
    return run


@router.patch("/issues/{issue_id}", response_model=IssueOut)
async def update_issue_status(
    issue_id: UUID,
    body: IssueUpdate,
    db: DbSession,
    user: CanResolveQuality,
) -> DataQualityIssue:
    from app.api.deps import is_super_admin

    # Verify issue belongs to caller's org (unless super-admin)
    issue_check = await db.get(DataQualityIssue, issue_id)
    if issue_check and not is_super_admin(user) and issue_check.org_id != user.org_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Issue not found")
    issue = await acknowledge_issue(db, issue_id, body.status, user.id)
    if issue is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Issue not found")
    return issue
