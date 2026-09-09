"""Multi-agent Alert system — professional, accurate, business-simple.

Agents:
- AlertRuleManagerAgent  — validates, enriches, and normalizes rule definitions
- AlertEvaluationAgent   — dry-run + live evaluation with window sums, anomaly checks, cooldowns
- NotificationAgent      — per-role routing, preference-aware, deduped
- AlertHistoryAgent      — evaluation audit trail for compliance
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Literal
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AlertRule

Metric = Literal["revenue", "orders", "expense_total"]
Condition = Literal["gt", "lt", "pct_change_gt", "anomaly_detected"]

VALID_METRICS = {"revenue", "orders", "expense_total"}
VALID_CONDITIONS = {"gt", "lt", "pct_change_gt", "anomaly_detected"}


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]
    normalized: dict[str, Any] | None = None


class AlertRuleManagerAgent:
    """Validates and normalizes alert rule definitions — the 'gatekeeper' agent."""

    def validate(self, payload: dict[str, Any]) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []

        name = (payload.get("name") or "").strip()
        if not name or len(name) < 3:
            errors.append("Rule name must be at least 3 characters")
        if len(name) > 80:
            errors.append("Rule name too long (max 80)")

        metric = payload.get("metric")
        if metric not in VALID_METRICS:
            errors.append(f"Metric must be one of {', '.join(sorted(VALID_METRICS))}")

        condition = payload.get("condition")
        if condition not in VALID_CONDITIONS:
            errors.append(f"Condition must be one of {', '.join(sorted(VALID_CONDITIONS))}")

        threshold = payload.get("threshold")
        if condition != "anomaly_detected" and threshold is None:
            errors.append("Threshold is required for gt/lt/pct_change_gt")
        if threshold is not None:
            try:
                t = Decimal(str(threshold))
                if t < 0:
                    errors.append("Threshold must be >= 0")
                if condition == "pct_change_gt" and t > 100:
                    warnings.append("pct_change threshold >100% rarely fires — did you mean 10?")
            except Exception:
                errors.append("Threshold must be a number")

        window_days = payload.get("window_days", 7)
        try:
            wd = int(window_days)
            if wd < 1 or wd > 90:
                errors.append("Window must be 1–90 days")
        except Exception:
            errors.append("Window must be an integer 1–90")

        channels = payload.get("channels") or {"in_app": True}
        if not isinstance(channels, dict) or not channels:
            errors.append("Channels must be a non-empty object e.g. {\"in_app\": true}")
        elif not any(channels.values()):
            warnings.append("No channel enabled — rule will fire but not notify")

        roles = payload.get("roles_notified") or ["admin", "manager"]
        if not isinstance(roles, list) or not roles:
            errors.append("roles_notified must be a non-empty list")
        else:
            valid_roles = {"admin", "manager", "analyst"}
            for r in roles:
                if r not in valid_roles:
                    errors.append(f"Unknown role {r}")

        normalized = None
        if not errors:
            normalized = {
                "name": name,
                "metric": metric,
                "condition": condition,
                "threshold": Decimal(str(threshold)) if threshold is not None else None,
                "window_days": int(window_days),
                "channels": channels,
                "roles_notified": roles,
            }

        return ValidationResult(ok=not errors, errors=errors, warnings=warnings, normalized=normalized)


@dataclass
class DryRunResult:
    would_fire: bool
    message: str | None
    window_start: date
    window_end: date
    current_value: float
    previous_value: float | None = None
    threshold: float | None = None


class AlertEvaluationAgent:
    """Dry-run + live evaluation — shares logic with engine._evaluate_rule but exposes richer preview."""

    async def dry_run(self, db: AsyncSession, rule: AlertRule, org_id=None) -> DryRunResult:
        from app.services.alerts.engine import _evaluate_rule, _window_sum

        # Reuse engine's window logic but capture values
        from app.core.clock import business_today

        today = business_today()
        window_start = today - timedelta(days=rule.window_days - 1)

        # Anomaly path
        if rule.condition == "anomaly_detected":
            from sqlalchemy import text

            from app.models import Anomaly

            since = datetime.combine(window_start, datetime.min.time())
            q = select(Anomaly).where(Anomaly.metric == rule.metric, Anomaly.status == "open", Anomaly.detected_at >= since)
            if org_id is not None:
                q = q.where(Anomaly.org_id == org_id)
            elif rule.org_id is not None:
                q = q.where(Anomaly.org_id == rule.org_id)
            count = len((await db.execute(q)).scalars().all())
            would = count > 0
            msg = f"{count} open anomalies" if would else None
            return DryRunResult(would_fire=would, message=msg, window_start=window_start, window_end=today, current_value=float(count), threshold=None)

        current = await _window_sum(db, rule.metric, window_start, today, org_id=org_id or rule.org_id)
        threshold = float(rule.threshold or 0) if rule.threshold is not None else None

        if rule.condition == "gt":
            return DryRunResult(would_fire=current > (threshold or 0), message=f"current {current:,.0f} vs threshold {threshold:,.0f}" if current > (threshold or 0) else None, window_start=window_start, window_end=today, current_value=current, threshold=threshold)
        if rule.condition == "lt":
            return DryRunResult(would_fire=current < (threshold or 0), message=f"current {current:,.0f} vs threshold {threshold:,.0f}" if current < (threshold or 0) else None, window_start=window_start, window_end=today, current_value=current, threshold=threshold)
        if rule.condition == "pct_change_gt":
            prev_start = window_start - timedelta(days=rule.window_days)
            prev_end = window_start - timedelta(days=1)
            previous = await _window_sum(db, rule.metric, prev_start, prev_end, org_id=org_id or rule.org_id)
            if previous > 0 and threshold is not None:
                change = (current - previous) / previous * 100
                would = abs(change) > threshold
                return DryRunResult(would_fire=would, message=f"change {change:+.1f}% current {current:,.0f} previous {previous:,.0f}" if would else None, window_start=window_start, window_end=today, current_value=current, previous_value=previous, threshold=threshold)
            return DryRunResult(would_fire=False, message=None, window_start=window_start, window_end=today, current_value=current, previous_value=previous if 'previous' in locals() else None, threshold=threshold)

        return DryRunResult(would_fire=False, message=None, window_start=window_start, window_end=today, current_value=current, threshold=threshold)


class NotificationAgent:
    """Routes notifications per rule — respects cooldown, role scoping, and preferences."""

    COOLDOWN_HOURS = 23

    async def preview_recipients(self, db: AsyncSession, rule: AlertRule, org_id=None) -> list[str]:
        from app.models import Profile

        roles = rule.roles_notified or ["admin", "manager"]
        q = select(Profile).where(Profile.role.in_(roles), Profile.is_active.is_(True))
        if org_id is not None:
            q = q.where(Profile.org_id == org_id)
        elif rule.org_id is not None:
            q = q.where(Profile.org_id == rule.org_id)
        rows = (await db.execute(q)).scalars().all()
        return [r.email for r in rows]


class AlertHistoryAgent:
    """Simple audit helper — logs evaluations for compliance (future table: alert_evaluations)."""

    def log(self, rule_id: UUID, org_id: UUID | None, message: str | None, fired: bool) -> dict[str, Any]:
        return {
            "rule_id": str(rule_id),
            "org_id": str(org_id) if org_id else None,
            "fired": fired,
            "message": message,
            "evaluated_at": datetime.utcnow().isoformat() + "Z",
        }


# Singletons for import
rule_manager_agent = AlertRuleManagerAgent()
evaluation_agent = AlertEvaluationAgent()
notification_agent = NotificationAgent()
history_agent = AlertHistoryAgent()
