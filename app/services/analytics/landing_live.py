"""Live platform metrics for the public landing page.

Everything here is a **whole-warehouse aggregate** — counts, monthly rollups and
share percentages. No customer, product-level or transaction-level rows are
returned, because ``GET /landing/live`` is unauthenticated. Keep it that way:
if a field would identify a customer or a single order, it does not belong here.
"""

from datetime import date, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import Date, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.clock import business_now, business_today
from app.models import (
    Anomaly,
    Customer,
    DataSource,
    EtlJob,
    Expense,
    Forecast,
    Insight,
    InventoryLevel,
    KpiSnapshot,
    MlModel,
    Product,
    SalesTransaction,
)

# How much history the landing visuals show.
SERIES_MONTHS = 12
FORECAST_MONTHS = 3
# The forecast table holds one series per target; the landing chart tracks revenue.
FORECAST_TARGET = "revenue_daily"


def _f(value: Any) -> float:
    """Decimal/None → float, so the response is plain JSON."""
    if value is None:
        return 0.0
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def _month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


async def _scalar(db: AsyncSession, stmt) -> Any:
    return (await db.execute(stmt)).scalar()


async def _totals(db: AsyncSession) -> dict[str, Any]:
    orders = await _scalar(db, select(func.count(SalesTransaction.id)))
    revenue = await _scalar(db, select(func.coalesce(func.sum(SalesTransaction.total_amount), 0)))
    expenses_rows = await _scalar(db, select(func.count(Expense.id)))
    expenses_total = await _scalar(db, select(func.coalesce(func.sum(Expense.amount), 0)))
    inventory_rows = await _scalar(db, select(func.count(InventoryLevel.id)))

    return {
        # "records unified" is the headline ETL number: every fact row the
        # pipelines have landed in the warehouse.
        "records_unified": int(orders or 0) + int(expenses_rows or 0) + int(inventory_rows or 0),
        "orders": int(orders or 0),
        "revenue": _f(revenue),
        "expenses": _f(expenses_total),
        "products": int(await _scalar(db, select(func.count(Product.id))) or 0),
        "customers": int(await _scalar(db, select(func.count(Customer.id))) or 0),
        "data_sources": int(await _scalar(db, select(func.count(DataSource.id))) or 0),
        "etl_jobs": int(await _scalar(db, select(func.count(EtlJob.id))) or 0),
        "kpi_points": int(await _scalar(db, select(func.count(KpiSnapshot.id))) or 0),
        "forecast_points": int(await _scalar(db, select(func.count(Forecast.id))) or 0),
        "models_trained": int(await _scalar(db, select(func.count(MlModel.id))) or 0),
        "anomalies_total": int(await _scalar(db, select(func.count(Anomaly.id))) or 0),
        "anomalies_open": int(
            await _scalar(db, select(func.count(Anomaly.id)).where(Anomaly.status == "open")) or 0
        ),
        "insights": int(await _scalar(db, select(func.count(Insight.id))) or 0),
    }


async def _coverage(db: AsyncSession) -> dict[str, str | None]:
    row = (
        await db.execute(
            select(func.min(SalesTransaction.txn_date), func.max(SalesTransaction.txn_date))
        )
    ).one()
    return {
        "from": row[0].isoformat() if row[0] else None,
        "to": row[1].isoformat() if row[1] else None,
    }


def _month_end(d: date) -> date:
    return (date(d.year + d.month // 12, d.month % 12 + 1, 1)) - timedelta(days=1)


async def _revenue_series(
    db: AsyncSession, latest: date | None
) -> list[dict[str, Any]]:
    """Monthly revenue/orders/expenses for the last ``SERIES_MONTHS`` months of data.

    The most recent month is usually mid-flight, so it is flagged ``partial`` —
    charts must not draw it as a real drop.
    """
    if latest is None:
        return []
    # Walk back whole months from the first of the latest month.
    start = date(latest.year, latest.month, 1)
    for _ in range(SERIES_MONTHS - 1):
        start = (start - timedelta(days=1)).replace(day=1)

    sales_month = cast(func.date_trunc("month", SalesTransaction.txn_date), Date).label("month")
    sales = (
        await db.execute(
            select(
                sales_month,
                func.coalesce(func.sum(SalesTransaction.total_amount), 0).label("revenue"),
                func.count(SalesTransaction.id).label("orders"),
            )
            .where(SalesTransaction.txn_date >= start)
            .group_by(sales_month)
            .order_by(sales_month)
        )
    ).all()

    exp_month = cast(func.date_trunc("month", Expense.expense_date), Date).label("month")
    expenses = {
        _month_key(r.month): _f(r.amount)
        for r in (
            await db.execute(
                select(exp_month, func.coalesce(func.sum(Expense.amount), 0).label("amount"))
                .where(Expense.expense_date >= start)
                .group_by(exp_month)
            )
        ).all()
    }

    out: list[dict[str, Any]] = []
    for r in sales:
        key = _month_key(r.month)
        spend = expenses.get(key, 0.0)
        out.append(
            {
                "month": key,
                "revenue": _f(r.revenue),
                "orders": int(r.orders),
                "expenses": spend,
                "net": round(_f(r.revenue) - spend, 2),
                # true only for the trailing month when data stops before month end
                "partial": r.month.year == latest.year
                and r.month.month == latest.month
                and latest < _month_end(latest),
            }
        )
    return out


async def _forecast_series(db: AsyncSession, latest: date | None) -> list[dict[str, Any]]:
    """Monthly totals of the active revenue forecast, with its confidence band.

    Forecasts are daily, so a month's total is the sum of its daily points; the
    ``days`` count exposes months the horizon only partly covers.
    """
    month = cast(func.date_trunc("month", Forecast.forecast_date), Date).label("month")
    floor = latest or business_today()
    stmt = (
        select(
            month,
            func.coalesce(func.sum(Forecast.yhat), 0).label("yhat"),
            func.coalesce(func.sum(Forecast.yhat_lower), 0).label("lower"),
            func.coalesce(func.sum(Forecast.yhat_upper), 0).label("upper"),
            func.count(Forecast.id).label("days"),
        )
        .where(Forecast.target == FORECAST_TARGET, Forecast.forecast_date > floor)
        .group_by(month)
        .order_by(month)
        .limit(FORECAST_MONTHS)
    )
    rows = (await db.execute(stmt)).all()
    return [
        {
            "month": _month_key(r.month),
            "yhat": round(_f(r.yhat), 2),
            "lower": round(_f(r.lower), 2),
            "upper": round(_f(r.upper), 2),
            "days": int(r.days),
        }
        for r in rows
    ]


async def _dimension(db: AsyncSession, column, latest: date | None, limit: int = 4) -> list[dict]:
    """Top-N revenue share by a sales dimension over the trailing 90 days."""
    if latest is None:
        return []
    since = latest - timedelta(days=90)
    stmt = (
        select(column, func.coalesce(func.sum(SalesTransaction.total_amount), 0).label("revenue"))
        .where(SalesTransaction.txn_date >= since, column.isnot(None))
        .group_by(column)
        .order_by(func.sum(SalesTransaction.total_amount).desc())
    )
    rows = (await db.execute(stmt)).all()
    total = sum(_f(r.revenue) for r in rows) or 1.0
    return [
        {
            "key": str(r[0]),
            "revenue": _f(r.revenue),
            "share_pct": round(_f(r.revenue) / total * 100, 1),
        }
        for r in rows[:limit]
    ]


async def _headline_kpis(db: AsyncSession, latest: date | None) -> dict[str, Any]:
    """Trailing 30 days vs the 30 before it — the numbers the hero tiles show.

    Margin is revenue − expenses (net), not gross: ``products.unit_cost`` is
    unpopulated in this warehouse, so a COGS-based gross margin would just
    restate revenue.
    """
    if latest is None:
        return {}
    cur_from = latest - timedelta(days=29)
    prev_from, prev_to = cur_from - timedelta(days=30), cur_from - timedelta(days=1)

    async def window(d_from: date, d_to: date) -> dict[str, float]:
        r = (
            await db.execute(
                select(
                    func.coalesce(func.sum(SalesTransaction.total_amount), 0).label("revenue"),
                    func.count(SalesTransaction.id).label("orders"),
                    func.coalesce(func.avg(SalesTransaction.total_amount), 0).label("aov"),
                ).where(SalesTransaction.txn_date.between(d_from, d_to))
            )
        ).one()
        spend = await _scalar(
            db,
            select(func.coalesce(func.sum(Expense.amount), 0)).where(
                Expense.expense_date.between(d_from, d_to)
            ),
        )
        return {
            "revenue": _f(r.revenue),
            "orders": float(r.orders),
            "aov": _f(r.aov),
            "expenses": _f(spend),
            "net": _f(r.revenue) - _f(spend),
        }

    cur = await window(cur_from, latest)
    prev = await window(prev_from, prev_to)

    def change(key: str) -> float | None:
        base = prev.get(key) or 0
        return round((cur[key] - base) / base * 100, 1) if base else None

    net_pct = round(cur["net"] / cur["revenue"] * 100, 1) if cur["revenue"] else None
    return {
        "window_days": 30,
        "period_start": cur_from.isoformat(),
        "period_end": latest.isoformat(),
        "revenue": round(cur["revenue"], 2),
        "revenue_change_pct": change("revenue"),
        "orders": int(cur["orders"]),
        "orders_change_pct": change("orders"),
        "avg_order_value": round(cur["aov"], 2),
        "avg_order_value_change_pct": change("aov"),
        "expenses": round(cur["expenses"], 2),
        "expenses_change_pct": change("expenses"),
        "net": round(cur["net"], 2),
        "net_change_pct": change("net"),
        "net_margin_pct": net_pct,
    }


async def _latest_anomaly(db: AsyncSession) -> dict[str, Any] | None:
    stmt = select(Anomaly).order_by(Anomaly.detected_at.desc()).limit(1)
    a = (await db.execute(stmt)).scalars().first()
    if a is None:
        return None
    return {
        "metric": a.metric,
        "severity": a.severity,
        "status": a.status,
        "observed_value": _f(a.observed_value),
        "expected_value": _f(a.expected_value) if a.expected_value is not None else None,
        "deviation_score": _f(a.deviation_score) if a.deviation_score is not None else None,
        "detected_at": a.detected_at.isoformat() if a.detected_at else None,
    }


async def _latest_insight(db: AsyncSession) -> dict[str, Any] | None:
    stmt = select(Insight).order_by(Insight.generated_at.desc()).limit(1)
    i = (await db.execute(stmt)).scalars().first()
    if i is None:
        return None
    return {
        "type": i.insight_type,
        "title": i.title,
        "body": i.body,
        "severity": i.severity,
        "generated_at": i.generated_at.isoformat() if i.generated_at else None,
    }


async def _active_model(db: AsyncSession) -> dict[str, Any] | None:
    """The model behind the landing forecast — revenue if trained, else newest."""
    stmt = (
        select(MlModel)
        .order_by(
            (MlModel.target == FORECAST_TARGET).desc(),
            MlModel.is_active.desc(),
            MlModel.trained_at.desc(),
        )
        .limit(1)
    )
    m = (await db.execute(stmt)).scalars().first()
    if m is None:
        return None
    return {
        "model_type": m.model_type,
        "target": m.target,
        "version": m.version,
        "training_rows": m.training_rows,
        "metrics": m.metrics or {},
        "is_active": m.is_active,
        "trained_at": m.trained_at.isoformat() if m.trained_at else None,
    }


async def _pipeline_health(db: AsyncSession) -> dict[str, Any]:
    rows = (
        await db.execute(select(EtlJob.status, func.count(EtlJob.id)).group_by(EtlJob.status))
    ).all()
    by_status = {str(s): int(n) for s, n in rows}
    total = sum(by_status.values()) or 1
    succeeded = by_status.get("succeeded", 0) + by_status.get("success", 0)
    last_run = await _scalar(db, select(func.max(EtlJob.finished_at)))
    return {
        "by_status": by_status,
        "success_rate_pct": round(succeeded / total * 100, 1),
        "last_run_at": last_run.isoformat() if last_run else None,
    }


async def build_live_metrics(db: AsyncSession) -> dict[str, Any]:
    """Assemble the whole public live-metrics payload."""
    coverage = await _coverage(db)
    latest = date.fromisoformat(coverage["to"]) if coverage["to"] else None

    return {
        "generated_at": business_now().isoformat(),
        "coverage": coverage,
        "totals": await _totals(db),
        "kpis": await _headline_kpis(db, latest),
        "revenue_series": await _revenue_series(db, latest),
        "forecast_series": await _forecast_series(db, latest),
        "regions": await _dimension(db, SalesTransaction.region, latest),
        "channels": await _dimension(db, SalesTransaction.channel, latest),
        "anomaly": await _latest_anomaly(db),
        "insight": await _latest_insight(db),
        "model": await _active_model(db),
        "pipeline": await _pipeline_health(db),
    }
