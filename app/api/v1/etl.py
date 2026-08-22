from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession, require_role
from app.models import DataSource, EtlJob
from app.schemas.integration import EtlJobOut

router = APIRouter(
    prefix="/etl", tags=["data-integration"], dependencies=[Depends(require_role("manager"))]
)


@router.post("/run/{source_id}", response_model=EtlJobOut)
async def run_source(
    source_id: UUID, db: DbSession, user: CurrentUser
) -> EtlJobOut:
    from app.api.deps import org_predicate, user_org_id

    source = (
        await db.execute(
            select(DataSource).where(
                DataSource.id == source_id, org_predicate(DataSource.org_id, user_org_id(user))
            )
        )
    ).scalar_one_or_none()
    if source is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Data source not found")
    if source.status != "active":
        raise HTTPException(status.HTTP_409_CONFLICT, f"Source is {source.status}")
    try:
        from app.services.etl.pipeline import run_source_pipeline

        result = await run_source_pipeline(db, source, trigger="manual")
    except ValueError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e)) from e
    job = await db.get(EtlJob, UUID(result.job_id))
    return EtlJobOut.model_validate(job)


@router.get("/jobs", response_model=list[EtlJobOut])
async def list_jobs(
    db: DbSession,
    user: CurrentUser,
    status_filter: str | None = Query(None, alias="status"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
) -> list[EtlJobOut]:
    from app.api.deps import org_predicate, user_org_id

    stmt = select(EtlJob).where(
        org_predicate(EtlJob.org_id, user_org_id(user))
    ).order_by(EtlJob.started_at.desc())
    if status_filter:
        stmt = stmt.where(EtlJob.status == status_filter)
    stmt = stmt.offset((page - 1) * page_size).limit(page_size)
    rows = (await db.execute(stmt)).scalars().all()
    return [EtlJobOut.model_validate(r) for r in rows]


@router.get("/jobs/{job_id}", response_model=EtlJobOut)
async def get_job(job_id: UUID, db: DbSession, user: CurrentUser) -> EtlJobOut:
    from app.api.deps import org_predicate, user_org_id

    job = (
        await db.execute(
            select(EtlJob).where(
                EtlJob.id == job_id, org_predicate(EtlJob.org_id, user_org_id(user))
            )
        )
    ).scalar_one_or_none()
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    return EtlJobOut.model_validate(job)
