"""APScheduler wiring — professional edition.

- Every cron tick is dispatched through the professional worker pool
  (`app.workers.pool`) so executions are durable, retried, and observable
  (BackgroundJob rows + in-memory metrics). This is what the All Transactions
  worker strip and `/admin/workers/*` read.
- Single-worker ownership is still enforced via the scheduler's own
  `max_instances=1` + Postgres advisory locks per source, but the pool's
  `SKIP LOCKED` claim loop makes the same code safe to run on N workers.
"""

import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import select, text

from app.core.clock import business_now, business_today
from app.core.database import get_session_factory
from app.models import DataSource
from app.workers import pool as worker_pool

logger = logging.getLogger(__name__)

_scheduler: AsyncIOScheduler | None = None


async def _run_scheduled_source(source_id: str) -> None:
    from app.services.etl.pipeline import run_source_pipeline

    async with get_session_factory()() as db:
        source = await db.get(DataSource, source_id)
        if source is None or source.status != "active":
            return
        # Advisory lock scoped by org_id + source_id so two orgs never block each other
        lock_key = f"{source.org_id}:{source_id}" if source.org_id else source_id
        lock = (await db.execute(text("SELECT pg_try_advisory_lock(hashtext(:key))"), {"key": lock_key})).scalar()
        if not lock:
            logger.warning("source %s already running; skipping", source_id)
            return
        try:
            result = await run_source_pipeline(db, source, trigger="schedule", org_id=source.org_id)
            logger.info("scheduled pull for %s: %s rows loaded", source.name, result.rows_loaded)
        except Exception:
            logger.exception("scheduled pull failed for source %s", source_id)
        finally:
            await db.execute(text("SELECT pg_advisory_unlock(hashtext(:key))"), {"key": lock_key})


# ── Pure business logic (no worker awareness) — kept separate so tests can call them directly
async def _weekly_retrain(org_id=None) -> None:
    from app.models import Organization
    from app.services.ml.registry import train_all

    if org_id is not None:
        async with get_session_factory()() as db:
            await train_all(db, org_id=org_id)
        return
    async with get_session_factory()() as db:
        orgs = (await db.execute(select(Organization).where(Organization.is_legacy.is_(False)))).scalars().all()
        # include legacy org as well for historical data
        all_orgs = orgs or []
        if not all_orgs:
            await train_all(db)
            return
        for org in all_orgs:
            async with get_session_factory()() as per_db:
                try:
                    await train_all(per_db, org_id=org.id)
                except Exception:
                    logger.exception("weekly retrain failed for org %s", org.id)


async def _daily_anomaly_scan(org_id=None) -> None:
    from app.services.ml.anomaly import scan_all

    if org_id is not None:
        async with get_session_factory()() as db:
            created = await scan_all(db, lookback_days=7, org_id=org_id)
            logger.info("daily anomaly scan org %s: %d new", org_id, created)
        return
    from app.models import Organization

    async with get_session_factory()() as db:
        orgs = (await db.execute(select(Organization))).scalars().all()
        if not orgs:
            async with get_session_factory()() as per_db:
                created = await scan_all(per_db, lookback_days=7)
                logger.info("daily anomaly scan: %d new", created)
            return
        for org in orgs:
            async with get_session_factory()() as per_db:
                try:
                    created = await scan_all(per_db, lookback_days=7, org_id=org.id)
                    logger.info("daily anomaly scan org %s: %d new", org.id, created)
                except Exception:
                    logger.exception("anomaly scan failed for org %s", org.id)


async def _daily_drift_check(org_id=None) -> None:
    from app.services.ml.drift import check_all

    if org_id is not None:
        async with get_session_factory()() as db:
            results = await check_all(db, org_id=org_id)
            triggered = sum(1 for r in results if r.triggered)
            logger.info("daily drift check org %s: %d model(s) checked, %d retrained", org_id, len(results), triggered)
        return
    from app.models import Organization

    async with get_session_factory()() as db:
        orgs = (await db.execute(select(Organization))).scalars().all()
        if not orgs:
            async with get_session_factory()() as per_db:
                results = await check_all(per_db)
                triggered = sum(1 for r in results if r.triggered)
                logger.info("daily drift check: %d model(s) checked, %d retrained", len(results), triggered)
            return
        for org in orgs:
            async with get_session_factory()() as per_db:
                try:
                    results = await check_all(per_db, org_id=org.id)
                    triggered = sum(1 for r in results if r.triggered)
                    logger.info(
                        "daily drift check org %s: %d model(s) checked, %d retrained", org.id, len(results), triggered
                    )
                except Exception:
                    logger.exception("drift check failed for org %s", org.id)


async def _nightly_insights_and_alerts(org_id=None) -> None:
    from app.services.alerts.engine import evaluate_alerts
    from app.services.insights.engine import generate_insights
    from app.services.ml.recommendations import persist_recommendations

    if org_id is not None:
        async with get_session_factory()() as db:
            insights = await generate_insights(db, org_id=org_id)
            notifications = await evaluate_alerts(db, org_id=org_id)
            recs = await persist_recommendations(db, org_id=org_id)
            logger.info(
                "nightly org %s: %d insights, %d notifications, %d/%d recommendations (new/generated)",
                org_id,
                insights,
                notifications,
                recs["new"],
                recs["generated"],
            )
        return
    from app.models import Organization

    async with get_session_factory()() as db:
        orgs = (await db.execute(select(Organization))).scalars().all()
        if not orgs:
            async with get_session_factory()() as per_db:
                insights = await generate_insights(per_db)
                notifications = await evaluate_alerts(per_db)
                recs = await persist_recommendations(per_db)
                logger.info(
                    "nightly: %d insights, %d notifications, %d/%d recommendations (new/generated)",
                    insights,
                    notifications,
                    recs["new"],
                    recs["generated"],
                )
            return
        for org in orgs:
            async with get_session_factory()() as per_db:
                try:
                    insights = await generate_insights(per_db, org_id=org.id)
                    notifications = await evaluate_alerts(per_db, org_id=org.id)
                    recs = await persist_recommendations(per_db, org_id=org.id)
                    logger.info(
                        "nightly org %s: %d insights, %d notifications, %d/%d recommendations",
                        org.id,
                        insights,
                        notifications,
                        recs["new"],
                        recs["generated"],
                    )
                except Exception:
                    logger.exception("nightly job failed for org %s", org.id)


# ── Worker-pool registrations — professional tracked execution (retries, metrics, durably logged)
# Each handler is a thin shim so the pool can retry/timeout/observe it independently.
@worker_pool.register("ml_retrain")
async def _w_ml_retrain(_payload: dict) -> None:
    await _weekly_retrain()


@worker_pool.register("anomaly_scan")
async def _w_anomaly_scan(_payload: dict) -> None:
    await _daily_anomaly_scan()


@worker_pool.register("drift_check")
async def _w_drift_check(_payload: dict) -> None:
    await _daily_drift_check()


@worker_pool.register("insights_alerts")
async def _w_insights_alerts(_payload: dict) -> None:
    await _nightly_insights_and_alerts()


@worker_pool.register("source_pull")
async def _w_source_pull(payload: dict) -> None:
    await _run_scheduled_source(str(payload["source_id"]))


@worker_pool.register("monthly_report")
async def _w_monthly_report(_payload: dict) -> None:
    await _monthly_report()


@worker_pool.register("quality_audit")
async def _w_quality_audit(_payload: dict) -> None:
    await _daily_quality_audit()


@worker_pool.register("report_schedules")
async def _w_report_schedules(_payload: dict) -> None:
    await _run_due_report_schedules()


@worker_pool.register("ai_retention_flush")
async def _w_ai_retention_flush(_payload: dict) -> None:
    await _daily_ai_retention_flush()


async def _monthly_report(org_id=None) -> None:
    from datetime import timedelta

    from app.models import Organization, Report
    from app.services.reports import builder
    from app.services.storage import FileStorage, make_key

    async def _generate_for_org(oid=None):
        async with get_session_factory()() as db:
            end = business_today().replace(day=1) - timedelta(days=1)
            start = end.replace(day=1)
            title = f"Monthly business summary — {start:%B %Y}"
            payload = await builder.build_pdf(db, start, end, title, org_id=oid)
            stored = FileStorage().save(
                make_key(f"monthly-{start:%Y-%m}-{oid}.pdf") if oid else make_key(f"monthly-{start:%Y-%m}.pdf"), payload
            )
            db.add(
                Report(
                    report_type="monthly_summary",
                    period_start=start,
                    period_end=end,
                    format="pdf",
                    s3_key=stored,
                    org_id=oid,
                )
            )
            await db.commit()
            logger.info("monthly report generated for %s org %s", start.strftime("%Y-%m"), oid)

    if org_id is not None:
        await _generate_for_org(org_id)
        return
    async with get_session_factory()() as db:
        orgs = (await db.execute(select(Organization))).scalars().all()
        if not orgs:
            await _generate_for_org(None)
            return
        for org in orgs:
            try:
                await _generate_for_org(org.id)
            except Exception:
                logger.exception("monthly report failed for org %s", org.id)


async def _daily_quality_audit(org_id=None) -> None:
    from app.services.quality.engine import run_quality_audit

    if org_id is not None:
        async with get_session_factory()() as db:
            run = await run_quality_audit(db, triggered_by="schedule", org_id=org_id)
            if run is None:
                logger.error("daily data-quality audit failed for org %s", org_id)
            else:
                logger.info("daily quality audit org %s: score=%s issues=%d", org_id, run.score, run.issues_found)
        return
    from app.models import Organization

    async with get_session_factory()() as db:
        orgs = (await db.execute(select(Organization))).scalars().all()
        if not orgs:
            async with get_session_factory()() as per_db:
                run = await run_quality_audit(per_db, triggered_by="schedule")
                if run is None:
                    logger.error("daily data-quality audit failed")
                else:
                    logger.info("daily quality audit: score=%s issues=%d", run.score, run.issues_found)
            return
        for org in orgs:
            async with get_session_factory()() as per_db:
                try:
                    run = await run_quality_audit(per_db, triggered_by="schedule", org_id=org.id)
                    if run is None:
                        logger.error("daily data-quality audit failed for org %s", org.id)
                    else:
                        logger.info(
                            "daily quality audit org %s: score=%s issues=%d", org.id, run.score, run.issues_found
                        )
                except Exception:
                    logger.exception("quality audit failed for org %s", org.id)


async def _daily_ai_retention_flush() -> None:
    """Auto-flush AI history older than retention_days (admin-configurable)."""
    from app.services.ai.retention import flush_expired_conversations

    async with get_session_factory()() as db:
        try:
            deleted = await flush_expired_conversations(db)
            await db.commit()
            if deleted:
                logger.info("ai retention flush deleted %d conversations", deleted)
        except Exception:
            logger.exception("ai retention flush failed")
            await db.rollback()


async def _run_due_report_schedules() -> None:
    """Fire every user report schedule whose next_run_at has arrived.

    Runs the same builder as the manual "Generate report" action, storing the
    file and dropping an in-app notification with the download link's report
    id, then rolls next_run_at forward so today's run can't repeat tomorrow.
    Also sends an email with the report attached when SMTP is configured.
    """
    from datetime import timedelta

    from sqlalchemy import select as sa_select

    from app.models import Notification, Profile, Report, ReportSchedule
    from app.services.reports import builder
    from app.services.reports.schedule import compute_next_run
    from app.services.storage import FileStorage, make_key

    today = business_today()
    async with get_session_factory()() as db:
        due = (
            (
                await db.execute(
                    sa_select(ReportSchedule).where(
                        ReportSchedule.is_active.is_(True),
                        ReportSchedule.next_run_at <= today,
                    )
                )
            )
            .scalars()
            .all()
        )
        for schedule in due:
            try:
                end = today
                start = end - timedelta(days=6 if schedule.frequency == "weekly" else 29)
                title = f"{schedule.frequency.capitalize()} report — {start:%d %b} to {end:%d %b %Y}"
                if schedule.format == "pdf":
                    payload = await builder.build_pdf(db, start, end, title)
                else:
                    payload = await builder.build_xlsx(db, start, end)
                key = make_key(f"scheduled-{schedule.id}-{end}.{schedule.format}")
                stored = FileStorage().save(key, payload)
                report = Report(
                    report_type="weekly_summary" if schedule.frequency == "weekly" else "monthly_summary",
                    period_start=start,
                    period_end=end,
                    format=schedule.format,
                    s3_key=stored,
                    generated_by=schedule.created_by,
                )
                db.add(report)
                await db.flush()
                db.add(
                    Notification(
                        user_id=schedule.created_by,
                        title="Your scheduled report is ready",
                        body=f"{title} — download it from the Reports page.",
                    )
                )
                schedule.last_run_at = business_now().replace(tzinfo=None)
                schedule.last_report_id = report.id
                schedule.next_run_at = compute_next_run(
                    schedule.frequency,
                    schedule.day_of_week,
                    schedule.day_of_month,
                    today,
                    include_today=False,
                )
                await db.commit()
                logger.info("scheduled report %s generated for schedule %s", report.id, schedule.id)

                # Best-effort email delivery — don't fail the schedule if it errors.
                try:
                    owner = await db.get(Profile, schedule.created_by)
                    if owner and owner.is_active and owner.email:
                        # Honor weekly_digest preference for weekly schedules.
                        if schedule.frequency == "weekly" and (owner.preferences or {}).get("weekly_digest") is False:
                            logger.info("skipping weekly digest email for %s (opted out)", owner.email)
                        else:
                            from app.services.email.service import send_report_ready_email

                            mime = (
                                "application/pdf"
                                if schedule.format == "pdf"
                                else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                            fname = f"report-{start}-{end}.{schedule.format}"
                            await send_report_ready_email(
                                owner.email,
                                title,
                                str(start),
                                str(end),
                                schedule.format,
                                attachment=(fname, payload, mime),
                            )
                except Exception:
                    logger.exception("report email failed for schedule %s", schedule.id)
            except Exception:
                logger.exception("failed to run report schedule %s", schedule.id)
                await db.rollback()


# ── Pool-backed tick wrappers — professional cron dispatch (observable, retried)
async def _tick_ml_retrain() -> None:
    await worker_pool.submit_and_wait("ml_retrain")


async def _tick_anomaly_scan() -> None:
    await worker_pool.submit_and_wait("anomaly_scan")


async def _tick_drift_check() -> None:
    await worker_pool.submit_and_wait("drift_check")


async def _tick_insights_alerts() -> None:
    await worker_pool.submit_and_wait("insights_alerts")


async def _tick_monthly_report() -> None:
    await worker_pool.submit_and_wait("monthly_report")


async def _tick_quality_audit() -> None:
    await worker_pool.submit_and_wait("quality_audit")


async def _tick_report_schedules() -> None:
    await worker_pool.submit_and_wait("report_schedules")


async def _tick_ai_retention_flush() -> None:
    await worker_pool.submit_and_wait("ai_retention_flush")


async def _tick_source_pull(source_id: str) -> None:
    await worker_pool.submit_and_wait("source_pull", {"source_id": source_id})


async def start_scheduler() -> AsyncIOScheduler:
    global _scheduler
    # Tune pool concurrency from env (default 4, ideal for 1 vCPU; raise to 8 on 2 vCPU)
    import asyncio
    import os

    concurrency = int(os.getenv("WORKER_CONCURRENCY", "4"))
    worker_pool._semaphore = asyncio.Semaphore(concurrency)  # type: ignore[attr-defined]
    worker_pool.start_claim_loop()
    # Report queue worker for handling 100 concurrent exports professionally
    try:
        from app.services.reports.queue import start_report_claim_loop

        start_report_claim_loop()
    except Exception:
        logger.exception("failed to start report claim loop")

    _scheduler = AsyncIOScheduler(job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 300})
    # ML lifecycle jobs (docs/05-ml-plan.md): weekly retrain, daily anomaly scan
    _scheduler.add_job(_tick_ml_retrain, CronTrigger.from_crontab("0 2 * * 1"), id="ml-retrain")
    _scheduler.add_job(_tick_anomaly_scan, CronTrigger.from_crontab("30 1 * * *"), id="anomaly-scan")
    _scheduler.add_job(_tick_drift_check, CronTrigger.from_crontab("0 1 * * *"), id="drift-check")
    # decision-support jobs (Phase 5): insights+alerts after the anomaly scan
    _scheduler.add_job(_tick_insights_alerts, CronTrigger.from_crontab("0 3 * * *"), id="insights-alerts")
    # Phase 3: daily data-quality audit after the anomaly scan
    _scheduler.add_job(_tick_quality_audit, CronTrigger.from_crontab("0 2 * * *"), id="quality-audit")
    _scheduler.add_job(_tick_monthly_report, CronTrigger.from_crontab("0 4 1 * *"), id="monthly-report")
    _scheduler.add_job(_tick_report_schedules, CronTrigger.from_crontab("0 5 * * *"), id="report-schedules")
    # Auto-flush AI history daily at 02:15 Asia/Kathmandu (20:30 UTC) — respects admin retention_days
    _scheduler.add_job(_tick_ai_retention_flush, CronTrigger.from_crontab("15 2 * * *"), id="ai-retention-flush")
    async with get_session_factory()() as db:
        sources = (
            (
                await db.execute(
                    select(DataSource).where(
                        DataSource.schedule_cron.is_not(None),
                        DataSource.status == "active",
                        DataSource.kind.in_(["rest_api", "postgres"]),
                    )
                )
            )
            .scalars()
            .all()
        )
    for source in sources:
        try:
            trigger = CronTrigger.from_crontab(source.schedule_cron)
        except ValueError:
            logger.error("invalid cron '%s' on source %s", source.schedule_cron, source.name)
            continue
        _scheduler.add_job(
            _tick_source_pull,
            trigger,
            args=[str(source.id)],
            id=str(source.id),
            coalesce=True,
            max_instances=1,
            misfire_grace_time=600,
        )
    _scheduler.start()
    logger.info(
        "scheduler started (pool=%s concurrency=%d) with %d cron job(s) + %d source pull(s)",
        worker_pool.worker_id(),
        concurrency,
        len([j for j in _scheduler.get_jobs() if not j.id.startswith("src")]),
        len(sources),
    )
    return _scheduler


def stop_scheduler() -> None:
    global _scheduler
    worker_pool.stop_claim_loop()
    try:
        from app.services.reports.queue import stop_report_claim_loop

        stop_report_claim_loop()
    except Exception:
        pass
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
