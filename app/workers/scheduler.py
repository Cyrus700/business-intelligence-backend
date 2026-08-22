"""APScheduler wiring: scheduled pulls for cron-configured rest_api/postgres sources.

Single-worker deployment owns the scheduler (documented in docs/plan/phase-2 risk
table); a Postgres advisory lock guards against overlapping runs of the same source.
"""

import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import select, text

from app.core.clock import business_now, business_today
from app.core.database import get_session_factory
from app.models import DataSource

logger = logging.getLogger(__name__)

_scheduler: AsyncIOScheduler | None = None


async def _run_scheduled_source(source_id: str) -> None:
    from app.services.etl.pipeline import run_source_pipeline

    async with get_session_factory()() as db:
        lock = (
            await db.execute(
                text("SELECT pg_try_advisory_lock(hashtext(:key))"), {"key": source_id}
            )
        ).scalar()
        if not lock:
            logger.warning("source %s already running; skipping", source_id)
            return
        try:
            source = await db.get(DataSource, source_id)
            if source is None or source.status != "active":
                return
            result = await run_source_pipeline(db, source, trigger="schedule")
            logger.info("scheduled pull for %s: %s rows loaded", source.name, result.rows_loaded)
        except Exception:
            logger.exception("scheduled pull failed for source %s", source_id)
        finally:
            await db.execute(text("SELECT pg_advisory_unlock(hashtext(:key))"), {"key": source_id})


async def _weekly_retrain() -> None:
    from app.services.ml.registry import train_all

    async with get_session_factory()() as db:
        await train_all(db)


async def _daily_anomaly_scan() -> None:
    from app.services.ml.anomaly import scan_all

    async with get_session_factory()() as db:
        created = await scan_all(db, lookback_days=7)
        logger.info("daily anomaly scan: %d new", created)


async def _daily_drift_check() -> None:
    from app.services.ml.drift import check_all

    async with get_session_factory()() as db:
        results = await check_all(db)
        triggered = sum(1 for r in results if r.triggered)
        logger.info("daily drift check: %d model(s) checked, %d retrained", len(results), triggered)


async def _nightly_insights_and_alerts() -> None:
    from app.services.alerts.engine import evaluate_alerts
    from app.services.insights.engine import generate_insights
    from app.services.ml.recommendations import persist_recommendations

    async with get_session_factory()() as db:
        insights = await generate_insights(db)
        notifications = await evaluate_alerts(db)
        recs = await persist_recommendations(db)
        logger.info(
            "nightly: %d insights, %d notifications, %d/%d recommendations (new/generated)",
            insights,
            notifications,
            recs["new"],
            recs["generated"],
        )


async def _monthly_report() -> None:
    from datetime import timedelta

    from app.models import Report
    from app.services.reports import builder
    from app.services.storage import FileStorage, make_key

    async with get_session_factory()() as db:
        end = business_today().replace(day=1) - timedelta(days=1)
        start = end.replace(day=1)
        title = f"Monthly business summary — {start:%B %Y}"
        payload = await builder.build_pdf(db, start, end, title)
        stored = FileStorage().save(make_key(f"monthly-{start:%Y-%m}.pdf"), payload)
        db.add(
            Report(
                report_type="monthly_summary",
                period_start=start,
                period_end=end,
                format="pdf",
                s3_key=stored,
            )
        )
        await db.commit()
        logger.info("monthly report generated for %s", start.strftime("%Y-%m"))


async def _daily_quality_audit() -> None:
    from app.services.quality.engine import run_quality_audit

    async with get_session_factory()() as db:
        run = await run_quality_audit(db, triggered_by="schedule")
        if run is None:
            logger.error("daily data-quality audit failed")
        else:
            logger.info("daily quality audit: score=%s issues=%d", run.score, run.issues_found)


async def _run_due_report_schedules() -> None:
    """Fire every user report schedule whose next_run_at has arrived.

    Runs the same builder as the manual "Generate report" action, storing the
    file and dropping an in-app notification with the download link's report
    id, then rolls next_run_at forward so today's run can't repeat tomorrow.
    """
    from datetime import timedelta

    from sqlalchemy import select as sa_select

    from app.models import Notification, Report, ReportSchedule
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
            except Exception:
                logger.exception("failed to run report schedule %s", schedule.id)
                await db.rollback()


async def start_scheduler() -> AsyncIOScheduler:
    global _scheduler
    _scheduler = AsyncIOScheduler()
    # ML lifecycle jobs (docs/05-ml-plan.md): weekly retrain, daily anomaly scan
    _scheduler.add_job(_weekly_retrain, CronTrigger.from_crontab("0 2 * * 1"), id="ml-retrain")
    _scheduler.add_job(
        _daily_anomaly_scan, CronTrigger.from_crontab("30 1 * * *"), id="anomaly-scan"
    )
    _scheduler.add_job(
        _daily_drift_check, CronTrigger.from_crontab("0 1 * * *"), id="drift-check"
    )
    # decision-support jobs (Phase 5): insights+alerts after the anomaly scan,
    # monthly summary report on the 1st
    _scheduler.add_job(
        _nightly_insights_and_alerts, CronTrigger.from_crontab("0 3 * * *"), id="insights-alerts"
    )
    # Phase 3: daily data-quality audit after the anomaly scan
    _scheduler.add_job(
        _daily_quality_audit, CronTrigger.from_crontab("0 2 * * *"), id="quality-audit"
    )
    _scheduler.add_job(_monthly_report, CronTrigger.from_crontab("0 4 1 * *"), id="monthly-report")
    _scheduler.add_job(
        _run_due_report_schedules, CronTrigger.from_crontab("0 5 * * *"), id="report-schedules"
    )
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
        _scheduler.add_job(_run_scheduled_source, trigger, args=[str(source.id)], id=str(source.id))
    _scheduler.start()
    logger.info("scheduler started with %d source job(s)", len(_scheduler.get_jobs()))
    return _scheduler


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
