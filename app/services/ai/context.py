"""Live business-data snapshot used to ground AI answers in real numbers.

Fetches a compact slice of the analytics warehouse (KPIs, top products,
expense categories, low stock, forecast, anomalies) so both the LLM prompts
and the local deterministic engine answer from actual data.
"""

from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.clock import BUSINESS_TZ_NAME, business_now, business_today
from app.models.ml import Anomaly, Forecast, MlModel
from app.services.analytics.queries import (
    Filters,
    data_coverage,
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


async def build_business_context(db: AsyncSession, days: int = DEFAULT_WINDOW_DAYS, *, org_id=None, user=None) -> str:
    # Resolve org scope: per-business isolation, super-admin sees all
    if org_id is None and user is not None:
        org_id = None if getattr(user, "is_super_admin", False) else getattr(user, "org_id", None)
    today = business_today()
    window_start = today - timedelta(days=days - 1)
    f = Filters(date_from=window_start, date_to=today, org_id=org_id)
    lines: list[str] = []

    # Temporal anchor first. Without it the model has no idea what "today" is
    # (its own notion comes from training data) and silently misreads every
    # relative date the user gives it.
    now = business_now()
    lines.append(f"- Today is {today:%A %d %B %Y} ({today.isoformat()}), current time {now:%H:%M} {BUSINESS_TZ_NAME}.")
    lines.append(f"- The figures below cover {window_start.isoformat()} → {today.isoformat()}.")
    # Business identity — must be first-class so "what is my business name?" is answered correctly
    try:
        if org_id is not None:
            from app.models.identity import Organization

            org = await db.get(Organization, org_id)
            if org:
                lines.append(f"- Business / workspace name: **{org.name}** (org_id {org.id})")
            else:
                lines.append(f"- Business / workspace id: {org_id} (name not found)")
        elif user is not None and getattr(user, "is_super_admin", False):
            lines.append("- Business / workspace: Platform Super-Admin (sees all organizations)")
            # Live platform totals so even a snapshot fallback answers counting questions precisely
            try:
                from sqlalchemy import func as _func
                from sqlalchemy import select as _sel

                from app.models.identity import Organization as _Org

                total_orgs = (await db.execute(_sel(_func.count()).select_from(_Org))).scalar() or 0
                approved = (await db.execute(_sel(_func.count()).select_from(_Org).where(_Org.status == "approved"))).scalar() or 0
                pending = (await db.execute(_sel(_func.count()).select_from(_Org).where(_Org.status == "pending"))).scalar() or 0
                rejected = (await db.execute(_sel(_func.count()).select_from(_Org).where(_Org.status == "rejected"))).scalar() or 0
                lines.append(
                    f"- Platform businesses registered: **{total_orgs}** (approved {approved}, pending {pending}, rejected {rejected}) — live right now"
                )
            except Exception:
                pass
        if user is not None:
            lines.append(
                f"- Signed-in user: {getattr(user, 'email', 'unknown')} · role {getattr(user, 'role', 'unknown')}"
            )
    except Exception:
        pass
    # Everything appended above is calendar metadata, not warehouse data — the
    # "nothing loaded yet" check at the end measures growth past this point.
    header_lines = len(lines)

    try:
        coverage = await data_coverage(db, org_id=org_id)
        if coverage["first_date"]:
            lines.append(
                f"- Warehouse holds data from {coverage['first_date']} to "
                f"{coverage['last_date']} ({coverage['days_behind']} day(s) behind today). "
                "For any date outside that range the answer is 'not loaded', never zero."
            )
            if coverage["last_ingested_at"]:
                lines.append(f"- Last upload: {coverage['last_ingested_at']:%Y-%m-%d %H:%M}.")
    except Exception:
        pass

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
        levels = await inventory_levels(db, org_id=org_id)
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
                f"- Inventory: {len(levels)} SKUs, {total_units:,.0f} units on hand (latest snapshot){low_txt}"
            )
    except Exception:
        pass

    try:
        mq = select(MlModel).where(MlModel.target == "revenue_daily", MlModel.is_active.is_(True))
        if org_id is not None:
            mq = mq.where(MlModel.org_id == org_id)
        model = (await db.execute(mq.order_by(MlModel.trained_at.desc()))).scalar_one_or_none()
        if model:
            fq = (
                select(Forecast)
                .where(Forecast.model_id == model.id)
                .order_by(Forecast.forecast_date)
                .limit(DEFAULT_WINDOW_DAYS)
            )
            if org_id is not None:
                fq = fq.where(Forecast.org_id == org_id)
            rows = (await db.execute(fq)).scalars().all()
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
        aq = select(Anomaly).where(Anomaly.status == "open").order_by(Anomaly.detected_at.desc()).limit(10)
        if org_id is not None:
            aq = aq.where(Anomaly.org_id == org_id)
        anoms = (await db.execute(aq)).scalars().all()
        if anoms:
            top = anoms[:3]
            detail = ", ".join(f"{a.metric.replace('_', ' ')} ({npr(a.observed_value)})" for a in top)
            lines.append(f"- {len(anoms)} open anomaly alert(s); latest: {detail}")
    except Exception:
        pass

    if len(lines) == header_lines:
        lines.append("- No analytics data loaded yet — connect a data source to populate the dashboard.")
    return "\n".join(lines)
