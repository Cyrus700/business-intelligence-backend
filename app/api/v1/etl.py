from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select

from app.api.deps import DbSession, require_role
from app.models import DataSource, EtlJob
from app.schemas.integration import EtlJobOut

router = APIRouter(
    prefix="/etl", tags=["data-integration"], dependencies=[Depends(require_role("manager"))]
)


@router.post("/run/{source_id}", response_model=EtlJobOut)
async def run_source(source_id: UUID, db: DbSession) -> EtlJobOut:
    from app.services.etl.pipeline import run_source_pipeline

    source = await db.get(DataSource, source_id)
    if source is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Data source not found")
    if source.status != "active":
        raise HTTPException(status.HTTP_409_CONFLICT, f"Source is {source.status}")
    try:
        result = await run_source_pipeline(db, source, trigger="manual")
    except ValueError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e)) from e
    job = await db.get(EtlJob, UUID(result.job_id))
    return EtlJobOut.model_validate(job)


@router.get("/jobs", response_model=list[EtlJobOut])
async def list_jobs(
    db: DbSession,
    status_filter: str | None = Query(None, alias="status"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
) -> list[EtlJobOut]:
    stmt = select(EtlJob).order_by(EtlJob.started_at.desc())
    if status_filter:
        stmt = stmt.where(EtlJob.status == status_filter)
    stmt = stmt.offset((page - 1) * page_size).limit(page_size)
    rows = (await db.execute(stmt)).scalars().all()
    return [EtlJobOut.model_validate(r) for r in rows]


@router.get("/jobs/{job_id}", response_model=EtlJobOut)
async def get_job(job_id: UUID, db: DbSession) -> EtlJobOut:
    job = await db.get(EtlJob, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    return EtlJobOut.model_validate(job)
