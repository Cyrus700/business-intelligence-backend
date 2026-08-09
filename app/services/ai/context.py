"""Live business-data snapshot used to ground AI answers in real numbers.

Fetches a compact slice of the analytics warehouse (KPIs, top products,
expense categories, low stock, forecast, anomalies) so both the LLM prompts
and the local deterministic engine answer from actual data.
"""

from datetime import date, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.ml import Anomaly, Forecast, MlModel
from app.services.analytics.queries import (
    Filters,
    expenses_by_category,
    inventory_levels,
    kpi_summary,
    sales_by_dimension,
)

DEFAULT_WINDOW_DAYS = 30


def _indian_grouped(value: float | int) -> str:
    """Indian digit grouping, e.g. 12345678 -> 1,23,45,678."""
    sign = "-" if value < 0 else ""
    s = str(int(abs(round(value))))
    if len(s) <= 3:
        return f"{sign}{s}"
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return f"{sign}{','.join(parts)},{tail}"


def npr(value: float | None) -> str:
    """Nepali-rupee formatted currency, e.g. रू 1,23,45,678."""
    return f"रू {_indian_grouped(value)}" if value is not None else "—"


async def build_business_context(
    db: AsyncSession, days: int = DEFAULT_WINDOW_DAYS
) -> str:
    today = date.today()
    f = Filters(date_from=today - timedelta(days=days - 1), date_to=today)
    lines: list[str] = []

    try:
        cards = {c["metric"]: c for c in await kpi_summary(db, f)}
        for metric in ("revenue", "orders", "avg_order_value", "gross_margin", "expense_total"):
            c = cards.get(metric)
            if not c:
                continue
            label = metric.replace("_", " ")
            change = c.get("change_pct")
            trend = f" ({(change or 0):+.1f}% vs previous {days}d)" if change is not None else ""
            lines.append(f"- {label}: {npr(c['value'])}{trend}")
    except Exception:
        pass

    try:
        for p in (await sales_by_dimension(db, f, "product"))[:5]:
            prefix = f"- Top product: {p['key']} — {npr(p['revenue'])}"
            lines.append(f"{prefix} ({p['share_pct']}% of sales, {p['orders']} orders)")
    except Exception:
        pass

    try:
        for e in (await expenses_by_category(db, f))[:5]:
            prefix = f"- Expense category: {e['key']} — {npr(e['revenue'])}"
            lines.append(f"{prefix} ({e['share_pct']}% of expenses)")
    except Exception:
        pass

    try:
        levels = await inventory_levels(db)
        if levels:
            total_units = sum(float(r["quantity_on_hand"]) for r in levels)
            low_items = [r for r in levels if r["below_reorder"]]
            low_txt = (
                f"; {len(low_items)} product(s) below reorder level: "
                + ", ".join((r["product"] or r["sku"]) for r in low_items[:6])
                if low_items
                else "; no products below reorder level"
            )
            lines.append(
                f"- Inventory: {len(levels)} SKUs, {total_units:,.0f} units on hand "
                f"(latest snapshot){low_txt}"
            )
    except Exception:
        pass

    try:
        model = (
            await db.execute(
                select(MlModel)
                .where(MlModel.target == "revenue_daily", MlModel.is_active.is_(True))
                .order_by(MlModel.trained_at.desc())
            )
        ).scalar_one_or_none()
        if model:
            rows = (
                (
                    await db.execute(
                        select(Forecast)
                        .where(Forecast.model_id == model.id)
                        .order_by(Forecast.forecast_date)
                        .limit(DEFAULT_WINDOW_DAYS)
                    )
                )
                .scalars()
                .all()
            )
            if rows:
                total = sum(float(r.yhat) for r in rows)
                avg = total / len(rows)
                lines.append(
                    f"- {len(rows)}-day revenue forecast: {npr(total)} total, ~{npr(avg)}/day "
                    f"({model.model_type} v{model.version})"
                )
    except Exception:
        pass

    try:
        anoms = (
            (
                await db.execute(
                    select(Anomaly)
                    .where(Anomaly.status == "open")
                    .order_by(Anomaly.detected_at.desc())
                    .limit(10)
                )
            )
            .scalars()
            .all()
        )
        if anoms:
            top = anoms[:3]
            detail = ", ".join(
                f"{a.metric.replace('_', ' ')} ({npr(a.observed_value)})" for a in top
            )
            lines.append(
                f"- {len(anoms)} open anomaly alert(s); latest: {detail}"
            )
    except Exception:
        pass

    if not lines:
        return (
            "No analytics data loaded yet — connect a data source to populate the dashboard."
        )
    return "\n".join(lines)
