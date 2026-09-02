"""Automated insight generation — the report's "last mile" (§3.1 Phase 5).

Deterministic detectors read the warehouse + ML tables and emit plain-language
findings with structured evidence. Dedupe via insights.dedupe_key so re-runs
never duplicate. No LLMs: template text keeps every claim traceable to stored
numbers (R5 trust requirement).
"""

import logging
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Anomaly, Forecast, Insight, InventoryLevel, MlModel, Product

logger = logging.getLogger(__name__)

KPI_SHIFT_THRESHOLD_PCT = 15.0
METRIC_LABEL = {"revenue": "Revenue", "orders": "Orders", "expense_total": "Expenses"}


async def _window_sum(db: AsyncSession, metric: str, start: date, end: date, org_id=None) -> float:
    if org_id is not None:
        value = (
            await db.execute(
                text(
                    "SELECT COALESCE(SUM(value), 0) FROM kpi_snapshots "
                    "WHERE metric = :m AND dimensions = '{}'::jsonb "
                    "AND snapshot_date BETWEEN :s AND :e AND org_id = :org_id"
                ),
                {"m": metric, "s": start, "e": end, "org_id": str(org_id)},
            )
        ).scalar_one()
    else:
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


async def _latest_data_date(db: AsyncSession, org_id=None) -> date | None:
    if org_id is not None:
        value = (
            await db.execute(
                text("SELECT MAX(snapshot_date) FROM kpi_snapshots WHERE metric = 'revenue' AND org_id = :org_id"),
                {"org_id": str(org_id)},
            )
        ).scalar_one()
    else:
        value = (
            await db.execute(
                text("SELECT MAX(snapshot_date) FROM kpi_snapshots WHERE metric = 'revenue'")
            )
        ).scalar_one()
    return value


def _fmt(value: float) -> str:
    return f"NPR {value:,.0f}"


async def detect_kpi_shifts(db: AsyncSession, today: date, org_id=None) -> list[dict[str, Any]]:
    found = []
    for metric, label in METRIC_LABEL.items():
        cur_start = today - timedelta(days=6)
        prev_start, prev_end = today - timedelta(days=13), today - timedelta(days=7)
        current = await _window_sum(db, metric, cur_start, today, org_id=org_id)
        previous = await _window_sum(db, metric, prev_start, prev_end, org_id=org_id)
        if previous <= 0:
            continue
        change = (current - previous) / previous * 100
        if abs(change) < KPI_SHIFT_THRESHOLD_PCT:
            continue
        direction = "up" if change > 0 else "down"
        bad = (metric == "expense_total") == (change > 0)
        found.append(
            {
                "insight_type": "comparison",
                "severity": "warning" if bad else "info",
                "title": f"{label} {direction} {abs(change):.0f}% week-over-week",
                "body": (
                    f"{label} for {cur_start:%b %d}–{today:%b %d} came to {_fmt(current)}, "
                    f"{direction} {abs(change):.0f}% from {_fmt(previous)} the week before. "
                    + (
                        "Worth reviewing what drove the increase."
                        if bad and direction == "up"
                        else "Investigate the drop against staffing, stock and channel performance."
                        if bad
                        else "Keep an eye on whether the momentum holds."
                    )
                ),
                "evidence": {
                    "metric": metric,
                    "current_week": current,
                    "previous_week": previous,
                    "change_pct": round(change, 1),
                },
                "period_start": cur_start,
                "period_end": today,
                "dedupe_key": f"kpi_shift:{metric}:{today.isoformat()}",
            }
        )
    return found


async def detect_forecast_outlook(db: AsyncSession, today: date, org_id=None) -> list[dict[str, Any]]:
    q = select(MlModel).where(MlModel.target == "revenue_daily", MlModel.is_active.is_(True))
    if org_id is not None:
        q = q.where(MlModel.org_id == org_id)
    model = (await db.execute(q)).scalar_one_or_none()
    if model is None:
        return []
    horizon_end = today + timedelta(days=30)
    fq = select(func.coalesce(func.sum(Forecast.yhat), 0)).where(
        Forecast.model_id == model.id,
        Forecast.forecast_date > today,
        Forecast.forecast_date <= horizon_end,
    )
    if org_id is not None:
        fq = fq.where(Forecast.org_id == org_id)
    forecast_sum = (await db.execute(fq)).scalar_one()
    actual_sum = await _window_sum(db, "revenue", today - timedelta(days=29), today, org_id=org_id)
    if actual_sum <= 0 or float(forecast_sum) <= 0:
        return []
    change = (float(forecast_sum) - actual_sum) / actual_sum * 100
    direction = "grow" if change > 0 else "decline"
    mape = (model.metrics or {}).get("mape")
    return [
        {
            "insight_type": "forecast",
            "severity": "info" if change >= 0 else "warning",
            "title": f"Revenue projected to {direction} {abs(change):.0f}% over the next 30 days",
            "body": (
                f"The {model.model_type} model projects {_fmt(float(forecast_sum))} in revenue "
                f"for the next 30 days, versus {_fmt(actual_sum)} over the last 30 "
                f"({'+' if change >= 0 else ''}{change:.0f}%). "
                f"Model accuracy on the 90-day holdout: MAPE {mape}%."
            ),
            "evidence": {
                "forecast_30d": float(forecast_sum),
                "actual_last_30d": actual_sum,
                "change_pct": round(change, 1),
                "model": f"{model.model_type} v{model.version}",
            },
            "period_start": today,
            "period_end": horizon_end,
            "dedupe_key": f"forecast_outlook:revenue:{today.isoformat()}",
        }
    ]


async def detect_new_anomalies(db: AsyncSession, today: date, org_id=None) -> list[dict[str, Any]]:
    since = datetime.combine(today - timedelta(days=1), datetime.min.time())
    q = select(Anomaly).where(Anomaly.status == "open", Anomaly.detected_at >= since)
    if org_id is not None:
        q = q.where(Anomaly.org_id == org_id)
    anomalies = ((await db.execute(q)).scalars().all())
    found = []
    for a in anomalies:
        context = a.context or {}
        label = METRIC_LABEL.get(a.metric, a.metric)
        found.append(
            {
                "insight_type": "anomaly",
                "severity": "critical" if a.severity == "high" else "warning",
                "title": f"{label} anomaly on {context.get('date', '?')}",
                "body": (
                    f"{label} was {_fmt(float(a.observed_value))} against an expected "
                    f"{_fmt(float(a.expected_value or 0))} "
                    f"({context.get('pct_deviation', '?')}% "
                    f"{context.get('direction', '')} normal). "
                    "Review the day's transactions and acknowledge the alert once triaged."
                ),
                "evidence": {"anomaly_id": str(a.id), **context},
                "related_anomaly_id": a.id,
                "period_start": date.fromisoformat(context["date"])
                if context.get("date")
                else None,
                "period_end": date.fromisoformat(context["date"]) if context.get("date") else None,
                "dedupe_key": f"anomaly:{a.id}",
            }
        )
    return found


async def detect_restock_recommendations(db: AsyncSession, today: date, org_id=None) -> list[dict[str, Any]]:
    latest_q = select(
        InventoryLevel.product_id,
        func.max(InventoryLevel.snapshot_date).label("latest_date"),
    )
    if org_id is not None:
        latest_q = latest_q.where(InventoryLevel.org_id == org_id)
    latest = latest_q.group_by(InventoryLevel.product_id).subquery()
    inv_cond = [InventoryLevel.quantity_on_hand <= InventoryLevel.reorder_level]
    if org_id is not None:
        inv_cond.append(InventoryLevel.org_id == org_id)
        prod_join = (Product.id == InventoryLevel.product_id) & (Product.org_id == org_id)
    else:
        prod_join = Product.id == InventoryLevel.product_id
    rows = (
        await db.execute(
            select(
                Product.sku,
                Product.name,
                InventoryLevel.quantity_on_hand,
                InventoryLevel.reorder_level,
            )
            .select_from(
                InventoryLevel.__table__.join(
                    latest,
                    (latest.c.product_id == InventoryLevel.product_id)
                    & (latest.c.latest_date == InventoryLevel.snapshot_date),
                ).join(Product.__table__, prod_join)
            )
            .where(*inv_cond)
        )
    ).all()
    found = []
    for sku, name, on_hand, reorder in rows[:5]:
        if org_id is not None:
            avg_daily = (
                await db.execute(
                    text(
                        "SELECT COALESCE(AVG(qty), 0) FROM ("
                        "  SELECT txn_date, SUM(quantity) AS qty FROM sales_transactions st"
                        "  JOIN products p ON p.id = st.product_id WHERE p.sku = :sku"
                        "  AND st.org_id = :org_id AND txn_date >= :since GROUP BY txn_date) t"
                    ),
                    {"sku": sku, "org_id": str(org_id), "since": today - timedelta(days=30)},
                )
            ).scalar_one()
        else:
            avg_daily = (
                await db.execute(
                    text(
                        "SELECT COALESCE(AVG(qty), 0) FROM ("
                        "  SELECT txn_date, SUM(quantity) AS qty FROM sales_transactions st"
                        "  JOIN products p ON p.id = st.product_id WHERE p.sku = :sku"
                        "  AND txn_date >= :since GROUP BY txn_date) t"
                    ),
                    {"sku": sku, "since": today - timedelta(days=30)},
                )
            ).scalar_one()
        suggested = max(int(float(avg_daily) * 30), reorder)
        found.append(
            {
                "insight_type": "recommendation",
                "severity": "warning",
                "title": f"Restock {name}",
                "body": (
                    f"{name} ({sku}) is at {on_hand} units — at or below its reorder level of "
                    f"{reorder}. At the recent average of {float(avg_daily):.1f} units/day, "
                    f"consider reordering ≥ {suggested} units to cover the next 30 days."
                ),
                "evidence": {
                    "sku": sku,
                    "on_hand": on_hand,
                    "reorder_level": reorder,
                    "avg_daily_qty_30d": round(float(avg_daily), 1),
                    "suggested_order_qty": suggested,
                },
                "period_start": today,
                "period_end": today,
                "dedupe_key": f"restock:{sku}:{today.isoformat()}",
            }
        )
    return found


async def generate_insights(db: AsyncSession, org_id=None) -> int:
    """Run all detectors; insert deduped insights. Returns number created."""
    today = await _latest_data_date(db, org_id=org_id)
    if today is None:
        return 0
    findings: list[dict[str, Any]] = []
    for detector in (
        detect_kpi_shifts,
        detect_forecast_outlook,
        detect_new_anomalies,
        detect_restock_recommendations,
    ):
        try:
            findings.extend(await detector(db, today, org_id=org_id))
        except Exception:
            logger.exception("insight detector %s failed", detector.__name__)

    created = 0
    for finding in findings:
        # Ensure org_id is set and dedupe_key is per-org to avoid cross-org dedup collisions
        if org_id is not None:
            finding.setdefault("org_id", org_id)
            # scope dedupe_key by org
            if finding.get("dedupe_key"):
                finding["dedupe_key"] = f"{org_id}:{finding['dedupe_key']}"
        stmt = (
            pg_insert(Insight)
            .values(**finding)
            .on_conflict_do_nothing(index_elements=["dedupe_key"])
            .returning(Insight.id)
        )
        if (await db.execute(stmt)).scalar_one_or_none() is not None:
            created += 1
    await db.commit()
    logger.info("insight generation: %d new (of %d findings)", created, len(findings))
    return created
