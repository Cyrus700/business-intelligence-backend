"""Admin endpoints: scheduler, storage, security, system status."""

from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func, select, text

from app.api.deps import DbSession, require_role
from app.core.clock import business_now
from app.models import EtlJob, MlModel, Profile
from app.services.storage import LOCAL_ROOT, FileStorage

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_role("admin"))],
)


# ── Scheduler ──────────────────────────────────────────────────────────────


class SchedulerJobOut(BaseModel):
    id: str
    name: str
    next_run: str | None
    last_run: str | None
    status: str
    schedule: str
    trigger: str


class SchedulerStatusOut(BaseModel):
    running: bool
    jobs: list[SchedulerJobOut]
    timezone: str


@router.get("/scheduler/status", response_model=SchedulerStatusOut)
async def scheduler_status() -> SchedulerStatusOut:
    """Current APScheduler status and job list."""
    from app.workers.scheduler import _scheduler

    jobs_out: list[SchedulerJobOut] = []
    if _scheduler:
        for job in _scheduler.get_jobs():
            next_run = job.next_run_time.isoformat() if job.next_run_time else None
            jobs_out.append(
                SchedulerJobOut(
                    id=job.id,
                    name=job.name,
                    next_run=next_run,
                    last_run=None,
                    status="scheduled" if next_run else "paused",
                    schedule=str(job.trigger),
                    trigger="cron",
                )
            )
    return SchedulerStatusOut(
        running=_scheduler.running if _scheduler else False,
        jobs=jobs_out,
        timezone="Asia/Kathmandu",
    )


@router.post("/scheduler/trigger/{job_id}")
async def trigger_scheduler_job(job_id: str) -> dict[str, str]:
    """Manually trigger a scheduler job."""
    from app.workers.scheduler import _scheduler

    if not _scheduler:
        return {"status": "error", "message": "Scheduler not running"}
    job = _scheduler.get_job(job_id)
    if not job:
        return {"status": "error", "message": f"Job {job_id} not found"}
    job.modify(next_run_time=datetime.now())
    return {"status": "ok", "message": f"Job {job_id} triggered"}


@router.post("/scheduler/pause/{job_id}")
async def pause_scheduler_job(job_id: str) -> dict[str, str]:
    """Pause a scheduler job."""
    from app.workers.scheduler import _scheduler

    if not _scheduler:
        return {"status": "error", "message": "Scheduler not running"}
    job = _scheduler.get_job(job_id)
    if not job:
        return {"status": "error", "message": f"Job {job_id} not found"}
    job.pause()
    return {"status": "ok", "message": f"Job {job_id} paused"}


@router.post("/scheduler/resume/{job_id}")
async def resume_scheduler_job(job_id: str) -> dict[str, str]:
    """Resume a paused scheduler job."""
    from app.workers.scheduler import _scheduler

    if not _scheduler:
        return {"status": "error", "message": "Scheduler not running"}
    job = _scheduler.get_job(job_id)
    if not job:
        return {"status": "error", "message": f"Job {job_id} not found"}
    job.resume()
    return {"status": "ok", "message": f"Job {job_id} resumed"}


# ── Storage ────────────────────────────────────────────────────────────────


class StorageInfoOut(BaseModel):
    backend: str
    bucket: str | None = None
    root_path: str | None = None
    total_files: int
    total_size_bytes: int
    usage_by_type: dict[str, dict[str, int]]
    recent_uploads: list[dict[str, Any]]


@router.get("/storage", response_model=StorageInfoOut)
async def storage_info(db: DbSession) -> StorageInfoOut:
    """Storage backend info, usage stats, and recent uploads."""
    fs = FileStorage()
    backend = "s3" if fs._use_s3 else "local"

    # Count files and sizes from uploads table
    from app.models import RawUpload

    total_files = (
        await db.execute(select(func.count()).select_from(RawUpload))
    ).scalar() or 0

    total_size = 0  # file_size column not available in RawUpload model

    # Usage by type (from raw_uploads.target_domain)
    type_stats = {}
    rows = (
        await db.execute(
            select(RawUpload.target_domain, func.count())
            .where(RawUpload.target_domain.is_not(None))
            .group_by(RawUpload.target_domain)
        )
    ).all()
    for domain, count in rows:
        type_stats[domain] = {"count": count, "size_bytes": 0}

    # Recent uploads
    recent = (
        await db.execute(
            select(RawUpload)
            .order_by(RawUpload.created_at.desc())
            .limit(20)
        )
    ).scalars().all()

    recent_uploads = [
        {
            "key": u.file_name,
            "size": 0,
            "type": u.target_domain or "unknown",
            "uploaded_at": u.created_at.isoformat(),
        }
        for u in recent
    ]

    return StorageInfoOut(
        backend=backend,
        bucket=fs._bucket if backend == "s3" else None,
        root_path=str(LOCAL_ROOT) if backend == "local" else None,
        total_files=total_files,
        total_size_bytes=total_size,
        usage_by_type=type_stats,
        recent_uploads=recent_uploads,
    )


# ── Security ───────────────────────────────────────────────────────────────


class SecurityStatusOut(BaseModel):
    ssrf: dict[str, Any]
    uploads: dict[str, Any]
    rate_limit: dict[str, Any]
    audit: dict[str, Any]
    auth: dict[str, Any]


@router.get("/security", response_model=SecurityStatusOut)
async def security_status(db: DbSession) -> SecurityStatusOut:
    """Security configuration and 24h statistics."""
    from app.core.config import get_settings
    from app.services.etl.ssrf import _BLOCKED_TLDS, _PRIVATE_NETS

    now = business_now()
    cutoff = (now - timedelta(hours=24)).replace(tzinfo=None)

    # SSRF stats (from audit logs where detail contains 'ssrf')
    ssrf_blocked = (
        await db.execute(
            select(func.count())
            .select_from(text("audit_logs"))
            .where(text("detail->>'ssrf_blocked' = 'true' AND created_at >= :cutoff")),
            {"cutoff": cutoff},
        )
    ).scalar() or 0

    # Upload rejections
    from app.models import RawUpload

    upload_rejected = (
        await db.execute(
            select(func.count())
            .select_from(RawUpload)
            .where(RawUpload.status == "rejected", RawUpload.created_at >= cutoff)
        )
    ).scalar() or 0

    # Failed logins
    failed_logins = (
        await db.execute(
            select(func.count())
            .select_from(text("audit_logs"))
            .where(text("action = 'auth.login_failed' AND created_at >= :cutoff")),
            {"cutoff": cutoff},
        )
    ).scalar() or 0

    # Audit events
    audit_events = (
        await db.execute(
            select(func.count())
            .select_from(text("audit_logs"))
            .where(text("created_at >= :cutoff")),
            {"cutoff": cutoff},
        )
    ).scalar() or 0

    # Rate limit current usage (approximate from request logs)
    # Simplified: just return config
    settings = get_settings()
    rate_limit_config = getattr(settings, "rate_limit_per_minute", 100)

    return SecurityStatusOut(
        ssrf={
            "enabled": True,
            "blocked_ranges": [str(n) for n in _PRIVATE_NETS],
            "blocked_tlds": list(_BLOCKED_TLDS),
            "blocked_count_24h": ssrf_blocked,
            "last_blocked": None,  # Could be enhanced to track
        },
        uploads={
            "enabled": True,
            "allowed_extensions": [".csv", ".xlsx", ".xls"],
            "max_size_mb": 50,
            "rejected_count_24h": upload_rejected,
            "last_rejected": None,
        },
        rate_limit={
            "enabled": True,
            "requests_per_minute": rate_limit_config,
            "current_usage_pct": min(
                100, (audit_events / max(1, rate_limit_config * 60 * 24)) * 100
            ),
        },
        audit={
            "enabled": True,
            "events_24h": audit_events,
            "last_event": None,
        },
        auth={
            "jwt_expiry_hours": 1,
            "bcrypt_cost": 12,
            "google_oauth": bool(settings.google_client_id),
            "failed_logins_24h": failed_logins,
            "last_failed": None,
        },
    )


# ── System quick stats ────────────────────────────────────────────────────


@router.get("/stats")
async def admin_stats(db: DbSession) -> dict[str, Any]:
    """Quick admin dashboard stats."""
    now = business_now()

    users = (await db.execute(select(func.count()).select_from(Profile))).scalar() or 0
    active_users = (
        await db.execute(
            select(func.count()).select_from(Profile).where(Profile.is_active.is_(True))
        )
    ).scalar() or 0
    etl_jobs = (await db.execute(select(func.count()).select_from(EtlJob))).scalar() or 0
    models = (await db.execute(select(func.count()).select_from(MlModel))).scalar() or 0
    active_models = (
        await db.execute(
            select(func.count()).select_from(MlModel).where(MlModel.is_active.is_(True))
        )
    ).scalar() or 0

    return {
        "users_total": users,
        "users_active": active_users,
        "etl_jobs_total": etl_jobs,
        "ml_models_total": models,
        "ml_models_active": active_models,
        "timestamp": now.isoformat(),
    }