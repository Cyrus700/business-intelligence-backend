"""Alert rule evaluation → notifications (in-app always; email when SMTP configured).

Conditions (docs/03-database-schema.md alert_rules):
  gt / lt           — window aggregate vs threshold
  pct_change_gt     — |window vs previous window| % change beyond threshold
  anomaly_detected  — any new open anomaly for the metric inside the window

A 23-hour per-rule cooldown prevents notification spam on scheduled re-evaluation.
"""

import logging
from datetime import date, datetime, timedelta

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.clock import business_now
from app.models import AlertRule, Anomaly, Notification, Profile

logger = logging.getLogger(__name__)

# Back-compat shim — older imports (e.g. tests) expect ``send_email`` here.
# Prefer ``from app.services.email import send_alert_email`` in new code.
try:
    from app.services.email.service import send_alert_email as _send_alert_email  # noqa: F401
except Exception:  # pragma: no cover
    _send_alert_email = None  # type: ignore

COOLDOWN_HOURS = 23


async def _window_sum(db: AsyncSession, metric: str, start: date, end: date) -> float:
    value = (
        await db.execute(
            text(
                "SELECT COALESCE(SUM(value), 0) FROM kpi_snapshots "
                "WHERE metric = :m AND dimensions = '{}'::jsonb "
                "AND snapshot_date BETWEEN :s AND :e"
            ),
            {"m": metric, "s": start, "e": end},
        )
    ).scalar_one()
    return float(value)


async def _evaluate_rule(db: AsyncSession, rule: AlertRule, today: date) -> str | None:
    """Returns the alert message when the rule fires, else None."""
    window_start = today - timedelta(days=rule.window_days - 1)
    if rule.condition == "anomaly_detected":
        since = datetime.combine(window_start, datetime.min.time())
        count = len(
            (
                await db.execute(
                    select(Anomaly).where(
                        Anomaly.metric == rule.metric,
                        Anomaly.status == "open",
                        Anomaly.detected_at >= since,
                    )
                )
            )
            .scalars()
            .all()
        )
        if count:
            noun = "anomalies" if count > 1 else "anomaly"
            return f"{count} open {noun} on {rule.metric} in the last {rule.window_days} days."
        return None

    current = await _window_sum(db, rule.metric, window_start, today)
    threshold = float(rule.threshold or 0)
    if rule.condition == "gt" and current > threshold:
        return (
            f"{rule.metric} over the last {rule.window_days} days is NPR {current:,.0f}, "
            f"above your NPR {threshold:,.0f} threshold."
        )
    if rule.condition == "lt" and current < threshold:
        return (
            f"{rule.metric} over the last {rule.window_days} days is NPR {current:,.0f}, "
            f"below your NPR {threshold:,.0f} threshold."
        )
    if rule.condition == "pct_change_gt":
        prev_start = window_start - timedelta(days=rule.window_days)
        prev_end = window_start - timedelta(days=1)
        previous = await _window_sum(db, rule.metric, prev_start, prev_end)
        if previous > 0:
            change = (current - previous) / previous * 100
            if abs(change) > threshold:
                return (
                    f"{rule.metric} changed {change:+.1f}% vs the previous {rule.window_days} days "
                    f"(NPR {current:,.0f} vs {previous:,.0f}), "
                    f"beyond your ±{threshold:.0f}% threshold."
                )
    return None


async def send_email(to: str, subject: str, body: str) -> bool:
    """Back-compat wrapper — delegates to the central email service.

    ``body`` is treated as the alert message; the rule name is extracted
    from the subject prefix ``[BI Dashboard] `` when present.
    """
    from app.services.email.service import send_alert_email

    rule_name = subject.removeprefix("[BI Dashboard] ").strip() or subject
    return await send_alert_email(to, rule_name, body)


async def evaluate_alerts(db: AsyncSession, today: date | None = None) -> int:
    """Evaluate all active rules; create notifications. Returns notifications created."""
    if today is None:
        today = (
            await db.execute(
                text("SELECT MAX(snapshot_date) FROM kpi_snapshots WHERE metric='revenue'")
            )
        ).scalar_one()
    if today is None:
        return 0

    rules = (
        (await db.execute(select(AlertRule).where(AlertRule.is_active.is_(True)))).scalars().all()
    )
    created = 0
    for rule in rules:
        cooldown_cutoff = business_now().replace(tzinfo=None) - timedelta(hours=COOLDOWN_HOURS)
        recent = (
            await db.execute(
                select(Notification)
                .where(
                    Notification.alert_rule_id == rule.id,
                    Notification.created_at >= cooldown_cutoff,
                )
                .limit(1)
            )
        ).scalar_one_or_none()
        if recent is not None:
            continue

        message = await _evaluate_rule(db, rule, today)
        if message is None:
            continue

        roles = rule.roles_notified or ["admin", "manager"]
        recipients = (
            (
                await db.execute(
                    select(Profile).where(Profile.role.in_(roles), Profile.is_active.is_(True))
                )
            )
            .scalars()
            .all()
        )
        for user in recipients:
            db.add(
                Notification(
                    user_id=user.id,
                    alert_rule_id=rule.id,
                    title=f"Alert: {rule.name}",
                    body=message,
                )
            )
            created += 1
            if (rule.channels or {}).get("email"):
                # Honor per-user preference (default on) and non-blocking send.
                prefs = (user.preferences or {})
                if prefs.get("anomaly_alerts") is False:
                    continue
                try:
                    # Keep alert evaluation non-blocking — email is best-effort
                    # and must never prevent the in-app notification from persisting.
                    from app.services.email.service import send_alert_email as _alert_send

                    await _alert_send(user.email, rule.name, message)
                except Exception:
                    logger.exception("email send failed for %s", user.email)
    await db.commit()
    logger.info("alert evaluation: %d notifications", created)
    return created
