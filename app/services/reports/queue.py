"""Report queue worker - handles 100 concurrent report exports professionally.

API enqueues a BackgroundJob with status pending, returns 202 immediately.
A dedicated claim loop (started with the scheduler) polls for pending
report_generate jobs, claims one with SKIP LOCKED, marks it claimed,
builds the report with concurrency limited to 2 via semaphore, creates
the Report row, and marks the job succeeded with report_id in payload.

UI polls GET /reports/jobs to show queue position, processing, completed,
and failed states. This keeps the event loop responsive even under 100
simultaneous requests, as each POST only does a single INSERT and returns.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, date, datetime
from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session_factory
from app.models.jobs import BackgroundJob

logger = logging.getLogger(__name__)

_report_sem = asyncio.Semaphore(2)
AVG_REPORT_SECONDS = 7


async def enqueue_report_job(
    db: AsyncSession,
    period_start: date,
    period_end: date,
    fmt: str,
    generated_by: UUID,
    user_email: str,
    email_me: bool,
    title: str,
) -> BackgroundJob:
    """Create a pending report_generate job and return it. Does not build."""
    payload = {
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "format": fmt,
        "generated_by": str(generated_by),
        "user_email": user_email,
        "email_me": email_me,
        "title": title,
    }
    job = BackgroundJob(name="report_generate", payload=payload, status="pending")
    db.add(job)
    await db.commit()
    await db.refresh(job)
    logger.info("enqueued report job %s for %s %s to %s", job.id, generated_by, period_start, period_end)
    return job


async def get_queue_position(job_id: UUID, db: AsyncSession) -> tuple[int, int]:
    """Return (position, total) for a report job, 1-indexed among pending+claimed."""
    job = await db.get(BackgroundJob, job_id)
    if not job or job.name != "report_generate":
        return 0, 0
    # Count pending jobs with earlier run_at (fifo)
    pending_before = (
        await db.execute(
            select(func.count())
            .select_from(BackgroundJob)
            .where(
                BackgroundJob.name == "report_generate",
                BackgroundJob.status == "pending",
                BackgroundJob.run_at <= job.run_at,
                BackgroundJob.id != job.id,
            )
        )
    ).scalar() or 0
    pending_total = (
        await db.execute(
            select(func.count())
            .select_from(BackgroundJob)
            .where(
                BackgroundJob.name == "report_generate",
                BackgroundJob.status == "pending",
            )
        )
    ).scalar() or 0
    processing = (
        await db.execute(
            select(func.count())
            .select_from(BackgroundJob)
            .where(
                BackgroundJob.name == "report_generate",
                BackgroundJob.status == "claimed",
            )
        )
    ).scalar() or 0
    # If this job is already claimed/processing, its position is processing count
    if job.status == "claimed":
        # Find its order among claimed
        claimed_before = (
            await db.execute(
                select(func.count())
                .select_from(BackgroundJob)
                .where(
                    BackgroundJob.name == "report_generate",
                    BackgroundJob.status == "claimed",
                    BackgroundJob.run_at <= job.run_at,
                )
            )
        ).scalar() or 0
        return int(claimed_before), int(pending_total + processing)
    if job.status == "pending":
        return int(pending_before + processing + 1), int(pending_total + processing)
    return 0, 0


async def _process_one_report_job(job: BackgroundJob) -> None:
    """Build a single report job. Called with the job already claimed (status=claimed)."""
    from app.models import Notification, Report
    from app.services.reports import builder
    from app.services.storage import FileStorage, make_key

    payload = job.payload or {}
    period_start = date.fromisoformat(str(payload["period_start"]))
    period_end = date.fromisoformat(str(payload["period_end"]))
    fmt = str(payload.get("format", "pdf"))
    generated_by = UUID(str(payload["generated_by"])) if payload.get("generated_by") else None
    email_me = bool(payload.get("email_me"))
    user_email = str(payload.get("user_email", ""))
    title = str(payload.get("title", f"Business summary {period_start:%d %b %Y} – {period_end:%d %b %Y}"))

    logger.info("report worker processing job %s for %s", job.id, generated_by)
    async with _report_sem:
        try:
            async with get_session_factory()() as db:
                if fmt == "pdf":
                    payload_bytes = await builder.build_pdf(db, period_start, period_end, title)
                else:
                    payload_bytes = await builder.build_xlsx(db, period_start, period_end)

                key = make_key(f"report-{period_start}-{period_end}-{job.id}.{fmt}")
                stored = FileStorage().save(key, payload_bytes)

                report = Report(
                    report_type="custom",
                    period_start=period_start,
                    period_end=period_end,
                    format=fmt,
                    s3_key=stored,
                    generated_by=generated_by,
                )
                db.add(report)
                await db.flush()
                report_id = report.id

                if generated_by:
                    db.add(
                        Notification(
                            user_id=generated_by,
                            title="Your report is ready",
                            body=f"{title} — download it from the Reports page.",
                        )
                    )
                await db.commit()
                logger.info("report job %s succeeded report %s", job.id, report_id)

            # Mark job succeeded and store report_id in payload
            async with get_session_factory()() as jdb:
                j = await jdb.get(BackgroundJob, job.id)
                if j:
                    j.status = "succeeded"
                    j.finished_at = datetime.now(UTC)
                    j.payload = {**(j.payload or {}), "report_id": str(report_id)}
                    await jdb.commit()

            if email_me and user_email:
                try:
                    from app.services.email.service import is_configured, send_report_ready_email

                    if not is_configured():
                        logger.warning("report %s SMTP not configured skip email %s", report_id, user_email)
                    else:
                        mime = (
                            "application/pdf"
                            if fmt == "pdf"
                            else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        fname = f"report-{period_start}-{period_end}.{fmt}"
                        await send_report_ready_email(
                            user_email,
                            title,
                            str(period_start),
                            str(period_end),
                            fmt,
                            attachment=(fname, payload_bytes, mime),
                        )
                except Exception:
                    logger.exception("report email failed job %s", job.id)

        except Exception as e:
            logger.exception("report job %s failed: %s", job.id, e)
            async with get_session_factory()() as jdb:
                j = await jdb.get(BackgroundJob, job.id)
                if j:
                    j.status = "failed"
                    j.last_error = f"{type(e).__name__}: {e}"
                    j.finished_at = datetime.now(UTC)
                    j.attempts = (j.attempts or 0) + 1
                    await jdb.commit()
            raise


async def claim_and_process_one() -> bool:
    """Try to claim one pending report job and process it. Returns True if a job was processed."""
    async with get_session_factory()() as db:
        # Use advisory lock so only one worker claims at a time (for multi-worker safety)
        has_lock = (await db.execute(text("SELECT pg_try_advisory_lock(hashtext('report-queue-claim'))"))).scalar()
        if not has_lock:
            return False
        try:
            row = (
                await db.execute(
                    select(BackgroundJob)
                    .where(BackgroundJob.name == "report_generate", BackgroundJob.status == "pending")
                    .order_by(BackgroundJob.run_at)
                    .limit(1)
                    .with_for_update(skip_locked=True)
                )
            ).scalar_one_or_none()
            if not row:
                return False
            row.status = "claimed"
            row.started_at = datetime.now(UTC)
            row.attempts = (row.attempts or 0) + 1
            await db.commit()
            job_id = row.id
        finally:
            await db.execute(text("SELECT pg_advisory_unlock(hashtext('report-queue-claim'))"))
            # Need to commit the unlock? The lock is session-level, need to keep session open? pg_try_advisory_lock is session-level, so unlocking in same session is fine.

    # Process outside the DB transaction, with semaphore limiting
    # Re-fetch the job in claimed status
    async with get_session_factory()() as cdb:
        job = await cdb.get(BackgroundJob, job_id)
        if not job:
            return False
    await _process_one_report_job(job)
    return True


_report_claim_task: asyncio.Task | None = None


async def _report_claim_loop(poll_interval: float = 1.0):
    while True:
        try:
            processed = await claim_and_process_one()
            if not processed:
                await asyncio.sleep(poll_interval)
                continue
            # If we processed one, immediately try next without sleep (drain queue)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("report claim loop error")
            await asyncio.sleep(poll_interval)


def start_report_claim_loop():
    global _report_claim_task
    if _report_claim_task is None or _report_claim_task.done():
        _report_claim_task = asyncio.create_task(_report_claim_loop(), name="report-claim-loop")
        logger.info("report claim loop started")


def stop_report_claim_loop():
    global _report_claim_task
    if _report_claim_task and not _report_claim_task.done():
        _report_claim_task.cancel()
