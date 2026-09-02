"""Decision-support endpoints: /insights, /alert-rules, /notifications, /reports (Phase 5)."""

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession, get_current_user, require_role
from app.core.clock import business_today
from app.models import AlertRule, Insight, Notification, Profile, Report, ReportSchedule
from app.services.reports.schedule import compute_next_run
from app.services.storage import FileStorage

router = APIRouter(tags=["decision-support"], dependencies=[Depends(get_current_user)])


# ---- insights ----


class InsightOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    insight_type: str
    title: str
    body: str
    severity: str
    evidence: dict[str, Any] | None
    period_start: date | None
    period_end: date | None
    generated_at: datetime
    is_pinned: bool


@router.get("/insights", response_model=list[InsightOut])
async def list_insights(
    db: DbSession,
    user: CurrentUser,
    insight_type: str | None = Query(None, alias="type"),
    severity: str | None = None,
    pinned: bool | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> list[InsightOut]:
    from app.api.deps import is_super_admin

    stmt = select(Insight).order_by(Insight.is_pinned.desc(), Insight.generated_at.desc())
    if not is_super_admin(user) and user.org_id is not None:
        stmt = stmt.where(Insight.org_id == user.org_id)
    if insight_type:
        stmt = stmt.where(Insight.insight_type == insight_type)
    if severity:
        stmt = stmt.where(Insight.severity == severity)
    if pinned is not None:
        stmt = stmt.where(Insight.is_pinned.is_(pinned))
    stmt = stmt.offset((page - 1) * page_size).limit(page_size)
    rows = (await db.execute(stmt)).scalars().all()
    return [InsightOut.model_validate(r) for r in rows]


@router.post(
    "/insights/generate",
    dependencies=[Depends(require_role("admin"))],
    status_code=status.HTTP_202_ACCEPTED,
)
async def generate_now(db: DbSession, user: CurrentUser) -> dict[str, int]:
    from app.services.insights.engine import generate_insights

    created = await generate_insights(db, org_id=user.org_id)
    return {"created": created}


@router.patch(
    "/insights/{insight_id}/pin",
    response_model=InsightOut,
    dependencies=[Depends(require_role("manager"))],
)
async def pin_insight(insight_id: UUID, db: DbSession, user: CurrentUser) -> InsightOut:
    from app.api.deps import is_super_admin

    insight = await db.get(Insight, insight_id)
    if insight is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Insight not found")
    if not is_super_admin(user) and insight.org_id != user.org_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Insight not found")
    insight.is_pinned = not insight.is_pinned
    await db.commit()
    await db.refresh(insight)
    return InsightOut.model_validate(insight)


# ---- alert rules ----


class AlertRuleIn(BaseModel):
    name: str
    metric: Literal["revenue", "orders", "expense_total"]
    condition: Literal["gt", "lt", "pct_change_gt", "anomaly_detected"]
    threshold: Decimal | None = None
    window_days: int = 7
    channels: dict[str, Any] = {"in_app": True}
    roles_notified: list[Literal["admin", "manager", "analyst"]] = ["admin", "manager"]


class AlertRuleUpdate(BaseModel):
    name: str | None = None
    threshold: Decimal | None = None
    window_days: int | None = None
    channels: dict[str, Any] | None = None
    roles_notified: list[str] | None = None
    is_active: bool | None = None


class AlertRuleOut(AlertRuleIn):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    is_active: bool
    created_at: datetime


manager_router = APIRouter(
    prefix="/alert-rules",
    tags=["decision-support"],
    dependencies=[Depends(require_role("manager"))],
)


@manager_router.get("", response_model=list[AlertRuleOut])
async def list_rules(db: DbSession, user: CurrentUser) -> list[AlertRuleOut]:
    from app.api.deps import is_super_admin

    stmt = select(AlertRule).order_by(AlertRule.created_at)
    if not is_super_admin(user) and user.org_id is not None:
        stmt = stmt.where(AlertRule.org_id == user.org_id)
    rows = (await db.execute(stmt)).scalars().all()
    return [AlertRuleOut.model_validate(r) for r in rows]


@manager_router.post("", response_model=AlertRuleOut, status_code=status.HTTP_201_CREATED)
async def create_rule(body: AlertRuleIn, db: DbSession, user: CurrentUser) -> AlertRuleOut:
    if body.condition != "anomaly_detected" and body.threshold is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "threshold is required for this condition")
    rule = AlertRule(**body.model_dump(), created_by=user.id, org_id=user.org_id)
    db.add(rule)
    await db.commit()
    await db.refresh(rule)
    return AlertRuleOut.model_validate(rule)


@manager_router.patch("/{rule_id}", response_model=AlertRuleOut)
async def update_rule(rule_id: UUID, body: AlertRuleUpdate, db: DbSession, user: CurrentUser) -> AlertRuleOut:
    from app.api.deps import is_super_admin

    rule = await db.get(AlertRule, rule_id)
    if rule is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Rule not found")
    if not is_super_admin(user) and rule.org_id != user.org_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Rule not found")
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(rule, field, value)
    await db.commit()
    await db.refresh(rule)
    return AlertRuleOut.model_validate(rule)


@manager_router.delete("/{rule_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_rule(rule_id: UUID, db: DbSession, user: CurrentUser) -> None:
    from app.api.deps import is_super_admin

    rule = await db.get(AlertRule, rule_id)
    if rule is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Rule not found")
    if not is_super_admin(user) and rule.org_id != user.org_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Rule not found")
    await db.delete(rule)
    await db.commit()


@manager_router.post("/evaluate", status_code=status.HTTP_202_ACCEPTED)
async def evaluate_now(db: DbSession, user: CurrentUser) -> dict[str, int]:
    from app.services.alerts.engine import evaluate_alerts

    created = await evaluate_alerts(db, org_id=user.org_id)
    return {"notifications": created}


# ---- notifications (own rows only) ----


class NotificationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    body: str | None
    is_read: bool
    created_at: datetime


@router.get("/notifications", response_model=list[NotificationOut])
async def my_notifications(
    db: DbSession,
    user: CurrentUser,
    unread_only: bool = False,
    page_size: int = Query(20, ge=1, le=100),
) -> list[NotificationOut]:
    stmt = (
        select(Notification)
        .where(Notification.user_id == user.id)
        .order_by(Notification.created_at.desc())
        .limit(page_size)
    )
    if unread_only:
        stmt = stmt.where(Notification.is_read.is_(False))
    rows = (await db.execute(stmt)).scalars().all()
    return [NotificationOut.model_validate(r) for r in rows]


@router.patch("/notifications/{notification_id}/read", response_model=NotificationOut)
async def mark_read(notification_id: UUID, db: DbSession, user: CurrentUser) -> NotificationOut:
    notification = await db.get(Notification, notification_id)
    if notification is None or notification.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Notification not found")
    notification.is_read = True
    await db.commit()
    await db.refresh(notification)
    return NotificationOut.model_validate(notification)


# ---- reports ----


class ReportOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    report_type: str
    period_start: date
    period_end: date
    format: str
    created_at: datetime


@router.get("/reports", response_model=list[ReportOut])
async def list_reports(db: DbSession, user: CurrentUser) -> list[ReportOut]:
    from app.api.deps import is_super_admin

    stmt = select(Report).order_by(Report.created_at.desc()).limit(50)
    if not is_super_admin(user) and user.org_id is not None:
        stmt = stmt.where(Report.org_id == user.org_id)
    rows = (await db.execute(stmt)).scalars().all()
    return [ReportOut.model_validate(r) for r in rows]


class ReportRequest(BaseModel):
    period_start: date
    period_end: date
    format: Literal["pdf", "xlsx"] = "pdf"
    email_me: bool = False  # when true, also email the file to the requester


class ReportJobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    status: str
    report_type: str | None = None
    period_start: date | None = None
    period_end: date | None = None
    format: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    report_id: UUID | None = None
    position: int | None = None
    total_in_queue: int | None = None
    estimated_wait_seconds: int | None = None
    error: str | None = None


@router.post(
    "/reports/generate",
    response_model=ReportJobOut,
    dependencies=[Depends(require_role("manager"))],
    status_code=status.HTTP_202_ACCEPTED,
)
async def generate_report(body: ReportRequest, db: DbSession, user: CurrentUser) -> ReportJobOut:
    """Enqueue a report build. Returns 202 with queue position. Worker builds it with concurrency 2.

    When 100 managers POST at once, each INSERT is fast and the worker drains the queue
    sequentially, so the API never blocks and the UI can show Queued → Processing → Completed.
    """
    if body.period_end < body.period_start:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "period_end before period_start")
    title = f"Business summary {body.period_start:%d %b %Y} – {body.period_end:%d %b %Y}"

    from app.services.reports.queue import enqueue_report_job

    job = await enqueue_report_job(
        db,
        period_start=body.period_start,
        period_end=body.period_end,
        fmt=body.format,
        generated_by=user.id,
        user_email=user.email,
        email_me=body.email_me,
        title=title,
    )

    # Return immediately - queue position will be polled via GET /reports/jobs
    return ReportJobOut(
        id=job.id,
        status=job.status,
        period_start=body.period_start,
        period_end=body.period_end,
        format=body.format,
        created_at=job.created_at if hasattr(job, "created_at") else None,
        report_id=None,
        position=None,
        total_in_queue=None,
        estimated_wait_seconds=None,
        error=None,
    )


@router.get("/reports/jobs", response_model=list[ReportJobOut])
async def list_report_jobs(
    db: DbSession, user: CurrentUser, status: str | None = None, limit: int = Query(20, ge=1, le=100)
) -> list[ReportJobOut]:
    """List report jobs — super-admin sees all, business admin sees own org's jobs, others see own only."""
    from sqlalchemy import select as sa_select

    from app.api.deps import is_super_admin
    from app.models.jobs import BackgroundJob
    from app.services.reports.queue import get_queue_position

    stmt = (
        sa_select(BackgroundJob)
        .where(BackgroundJob.name == "report_generate")
        .order_by(BackgroundJob.run_at.desc())
        .limit(limit)
    )
    if is_super_admin(user):
        pass  # super sees all
    elif user.role == "admin":
        # Business admin: see jobs of users in same org (via profiles)
        stmt = stmt.where(
            BackgroundJob.payload["generated_by"].astext.in_(sa_select(Profile.id).where(Profile.org_id == user.org_id))
        )
    else:
        # Manager/analyst: only own jobs
        stmt = stmt.where(BackgroundJob.payload["generated_by"].astext == str(user.id))
    if status:
        stmt = stmt.where(BackgroundJob.status == status)
    rows = (await db.execute(stmt)).scalars().all()
    out: list[ReportJobOut] = []
    for r in rows:
        payload = r.payload or {}
        pos = None
        total = None
        est = None
        if r.status in ("pending", "claimed"):
            try:
                pos, total = await get_queue_position(r.id, db)
                est = max(0, (pos - 1) * 7) if pos else 0
            except Exception:
                pass
        out.append(
            ReportJobOut(
                id=r.id,
                status=r.status,
                period_start=date.fromisoformat(payload["period_start"]) if payload.get("period_start") else None,
                period_end=date.fromisoformat(payload["period_end"]) if payload.get("period_end") else None,
                format=payload.get("format"),
                created_at=r.created_at if hasattr(r, "created_at") else r.run_at,
                started_at=r.started_at,
                finished_at=r.finished_at,
                report_id=UUID(payload["report_id"]) if payload.get("report_id") else None,
                position=pos,
                total_in_queue=total,
                estimated_wait_seconds=est,
                error=r.last_error,
            )
        )
    return out


@router.get("/reports/jobs/{job_id}", response_model=ReportJobOut)
async def get_report_job(job_id: UUID, db: DbSession, user: CurrentUser) -> ReportJobOut:
    from app.models.jobs import BackgroundJob
    from app.services.reports.queue import get_queue_position

    job = await db.get(BackgroundJob, job_id)
    if not job or job.name != "report_generate":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    payload = job.payload or {}
    # Only owner or admin can view
    if str(payload.get("generated_by")) != str(user.id) and user.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Not allowed")
    pos = None
    total = None
    est = None
    if job.status in ("pending", "claimed"):
        try:
            pos, total = await get_queue_position(job.id, db)
            est = max(0, (pos - 1) * 7) if pos else 0
        except Exception:
            pass
    return ReportJobOut(
        id=job.id,
        status=job.status,
        period_start=date.fromisoformat(payload["period_start"]) if payload.get("period_start") else None,
        period_end=date.fromisoformat(payload["period_end"]) if payload.get("period_end") else None,
        format=payload.get("format"),
        created_at=job.created_at if hasattr(job, "created_at") else job.run_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        report_id=UUID(payload["report_id"]) if payload.get("report_id") else None,
        position=pos,
        total_in_queue=total,
        estimated_wait_seconds=est,
        error=job.last_error,
    )


@router.get("/reports/{report_id}/download")
async def download_report(report_id: UUID, db: DbSession, user: CurrentUser) -> Response:
    from app.api.deps import is_super_admin

    report = await db.get(Report, report_id)
    if report is None or not report.s3_key:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Report not found")
    if not is_super_admin(user) and report.org_id != user.org_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Report not found")
    key = report.s3_key.split("var/uploads/")[-1] if "var/uploads/" in report.s3_key else report.s3_key
    payload = FileStorage().load(key)
    media = (
        "application/pdf"
        if report.format == "pdf"
        else ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    )
    filename = f"report-{report.period_start}-{report.period_end}.{report.format}"
    return Response(
        content=payload,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---- report schedules ----


class ReportScheduleIn(BaseModel):
    frequency: Literal["weekly", "monthly"]
    format: Literal["pdf", "xlsx"] = "pdf"
    day_of_week: int | None = None  # 0=Mon..6=Sun, required for weekly
    day_of_month: int | None = None  # 1..28, required for monthly

    def validate_day(self) -> None:
        if self.frequency == "weekly" and self.day_of_week is None:
            raise HTTPException(422, "day_of_week is required for a weekly schedule")
        if self.frequency == "monthly" and self.day_of_month is None:
            raise HTTPException(422, "day_of_month is required for a monthly schedule")
        if self.day_of_week is not None and not (0 <= self.day_of_week <= 6):
            raise HTTPException(422, "day_of_week must be 0-6")
        if self.day_of_month is not None and not (1 <= self.day_of_month <= 28):
            raise HTTPException(422, "day_of_month must be 1-28")


class ReportScheduleUpdate(BaseModel):
    format: Literal["pdf", "xlsx"] | None = None
    day_of_week: int | None = None
    day_of_month: int | None = None
    is_active: bool | None = None


class ReportScheduleOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    frequency: str
    format: str
    day_of_week: int | None
    day_of_month: int | None
    is_active: bool
    next_run_at: date
    last_run_at: datetime | None
    last_report_id: UUID | None
    created_at: datetime


schedule_router = APIRouter(
    prefix="/report-schedules",
    tags=["decision-support"],
    dependencies=[Depends(require_role("manager"))],
)


@schedule_router.get("", response_model=list[ReportScheduleOut])
async def list_schedules(db: DbSession, user: CurrentUser) -> list[ReportScheduleOut]:
    rows = (
        (
            await db.execute(
                select(ReportSchedule)
                .where(ReportSchedule.created_by == user.id)
                .order_by(ReportSchedule.created_at.desc())
            )
        )
        .scalars()
        .all()
    )
    return [ReportScheduleOut.model_validate(r) for r in rows]


@schedule_router.post("", response_model=ReportScheduleOut, status_code=status.HTTP_201_CREATED)
async def create_schedule(body: ReportScheduleIn, db: DbSession, user: CurrentUser) -> ReportScheduleOut:
    body.validate_day()
    today = business_today()
    schedule = ReportSchedule(
        frequency=body.frequency,
        format=body.format,
        day_of_week=body.day_of_week,
        day_of_month=body.day_of_month,
        next_run_at=compute_next_run(body.frequency, body.day_of_week, body.day_of_month, today),
        created_by=user.id,
        org_id=user.org_id,
    )
    db.add(schedule)
    await db.commit()
    await db.refresh(schedule)
    return ReportScheduleOut.model_validate(schedule)


@schedule_router.patch("/{schedule_id}", response_model=ReportScheduleOut)
async def update_schedule(
    schedule_id: UUID, body: ReportScheduleUpdate, db: DbSession, user: CurrentUser
) -> ReportScheduleOut:
    schedule = await db.get(ReportSchedule, schedule_id)
    if schedule is None or schedule.created_by != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Schedule not found")
    changes = body.model_dump(exclude_unset=True)
    for field, value in changes.items():
        setattr(schedule, field, value)
    if "day_of_week" in changes or "day_of_month" in changes:
        schedule.next_run_at = compute_next_run(
            schedule.frequency, schedule.day_of_week, schedule.day_of_month, business_today()
        )
    await db.commit()
    await db.refresh(schedule)
    return ReportScheduleOut.model_validate(schedule)


@schedule_router.delete("/{schedule_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_schedule(schedule_id: UUID, db: DbSession, user: CurrentUser) -> None:
    schedule = await db.get(ReportSchedule, schedule_id)
    if schedule is None or schedule.created_by != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Schedule not found")
    await db.delete(schedule)
    await db.commit()
