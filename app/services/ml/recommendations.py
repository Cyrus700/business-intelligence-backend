from datetime import date, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.clock import business_today


def _fmt(value: float) -> str:
    return f"NPR {value:,.0f}"


def _ninety_days_ago(today: date) -> date:
    return today - timedelta(days=90)


async def revenue_recommendations(db: AsyncSession, today: date) -> list[dict[str, Any]]:
    found = []
    thirty = today - timedelta(days=30)
    ninety = today - timedelta(days=90)

    ch = await db.execute(
        text("""
            SELECT channel, SUM(total_amount) AS revenue,
                   ROW_NUMBER() OVER (ORDER BY SUM(total_amount) DESC) AS rnk
            FROM sales_transactions
            WHERE txn_date BETWEEN :s AND :e
            GROUP BY channel
            ORDER BY revenue DESC
        """),
        {"s": thirty, "e": today},
    )
    channels = ch.all()
    if len(channels) >= 2:
        top_channel = channels[0]
        threshold = float(top_channel.revenue) * 0.3
        bottom_channels = [c for c in channels if float(c.revenue) < threshold]
        for bc in bottom_channels[:2]:
            gap_pct = round((1 - float(bc.revenue) / float(top_channel.revenue)) * 100, 1)
            top_value = _fmt(float(top_channel.revenue))
            found.append({
                "insight_type": "recommendation",
                "severity": "info",
                "title": f"Boost {bc.channel} channel revenue",
                "body": (
                    f"{bc.channel} generated {_fmt(float(bc.revenue))} in the last 30 days — "
                    f"{gap_pct}% behind {top_channel.channel} ({top_value}). "
                    "Consider targeted promotions or inventory allocation to close the gap."
                ),
                "evidence": {
                    "channel": bc.channel,
                    "revenue_30d": float(bc.revenue),
                    "top_channel": top_channel.channel,
                    "top_revenue_30d": float(top_channel.revenue),
                    "gap_pct": gap_pct,
                },
                "dedupe_key": f"channel_boost:{bc.channel}:{today.isoformat()}",
            })

    peak = await db.execute(
        text("""
            SELECT DATE_TRUNC('day', txn_date) AS day, SUM(total_amount) AS revenue
            FROM sales_transactions
            WHERE txn_date BETWEEN :s AND :e
            GROUP BY day ORDER BY revenue DESC LIMIT 1
        """),
        {"s": ninety, "e": today},
    )
    peak_row = peak.one_or_none()
    if peak_row:
        avg = await db.execute(
            text("""
                SELECT AVG(daily) FROM (
                    SELECT SUM(total_amount) AS daily
                    FROM sales_transactions
                    WHERE txn_date BETWEEN :s AND :e
                    GROUP BY txn_date
                ) d
            """),
            {"s": ninety, "e": today},
        )
        avg_val = avg.scalar_one() or 0
        if avg_val > 0 and float(peak_row.revenue) > avg_val * 2.5:
            found.append({
                "insight_type": "recommendation",
                "severity": "info",
                "title": f"Replicate peak day performance: {peak_row.day:%b %d}",
                "body": (
                    f"{peak_row.day:%b %d} generated {_fmt(float(peak_row.revenue))} — "
                    f"{round(float(peak_row.revenue) / avg_val, 1)}x the daily average. "
                    "Review what drove that day (promotions, marketing, events) and replicate."
                ),
                "evidence": {
                    "peak_date": peak_row.day.isoformat(),
                    "peak_revenue": float(peak_row.revenue),
                    "avg_daily_revenue_90d": round(float(avg_val), 2),
                    "multiplier": round(float(peak_row.revenue) / avg_val, 1),
                },
                "dedupe_key": f"peak_day:{today.isoformat()}",
            })

    return found


async def cost_recommendations(db: AsyncSession, today: date) -> list[dict[str, Any]]:
    found = []
    thirty = today - timedelta(days=30)

    top_exp = await db.execute(
        text("""
            SELECT category, SUM(amount) AS total
            FROM expenses
            WHERE expense_date BETWEEN :s AND :e
            GROUP BY category
            ORDER BY total DESC
            LIMIT 3
        """),
        {"s": thirty, "e": today},
    )
    expenses = top_exp.all()
    if expenses:
        total_exp = sum(float(e.total) for e in expenses)
        for e in expenses:
            share = round(float(e.total) / total_exp * 100, 1) if total_exp > 0 else 0
            if share > 30:
                found.append({
                    "insight_type": "recommendation",
                    "severity": "warning",
                    "title": f"{e.category} is {share}% of total expenses",
"body": (
                    f"{e.category} costs totaled {_fmt(float(e.total))} in the last 30 days "
                    f"({share}% of tracked expenses). Review for consolidation or "
                    "renegotiation opportunities."
                ),
                    "evidence": {
                        "category": e.category,
                        "amount_30d": float(e.total),
                        "share_pct": share,
                    },
                    "dedupe_key": f"cost_share:{e.category}:{today.isoformat()}",
                })

    exp_trend = await db.execute(
        text("""
            SELECT DATE_TRUNC('month', expense_date) AS month, SUM(amount) AS total
            FROM expenses
            WHERE expense_date >= :s
            GROUP BY month ORDER BY month
        """),
        {"s": _ninety_days_ago(today)},
    )
    months = exp_trend.all()
    if len(months) >= 2:
        latest = float(months[-1].total)
        prev = float(months[-2].total)
        if prev > 0:
            change = round((latest - prev) / prev * 100, 1)
            if change > 10:
                found.append({
                    "insight_type": "recommendation",
                    "severity": "warning",
                    "title": f"Expenses rose {change}% month-over-month",
                    "body": (
                        f"Monthly expenses increased from {_fmt(prev)} to {_fmt(latest)} "
                        f"({change}% MoM). Investigate the categories driving the increase."
                    ),
                    "evidence": {
                        "previous_month_total": prev,
                        "current_month_total": latest,
                        "change_pct": change,
                    },
                    "dedupe_key": f"cost_mom:{today.isoformat()}",
                })

    return found


async def pricing_recommendations(db: AsyncSession, today: date) -> list[dict[str, Any]]:
    found = []
    thirty = today - timedelta(days=30)

    deep_discount = await db.execute(
        text("""
            SELECT p.name, p.sku, AVG(st.unit_price) AS avg_price,
                   AVG(st.discount) AS avg_discount,
                   COUNT(*) AS txns, SUM(st.total_amount) AS revenue
            FROM sales_transactions st
            JOIN products p ON p.id = st.product_id
            WHERE st.txn_date BETWEEN :s AND :e
              AND st.discount > 0
            GROUP BY p.id, p.name, p.sku
            HAVING AVG(st.discount) > 20 AND COUNT(*) >= 10
            ORDER BY AVG(st.discount) DESC
            LIMIT 3
        """),
        {"s": thirty, "e": today},
    )
    for row in deep_discount.all():
        rev_impact = round(float(row.revenue) / float(row.txns), 2)
        found.append({
            "insight_type": "recommendation",
            "severity": "info",
            "title": f"High discounts on {row.name}",
            "body": (
                f"{row.name} ({row.sku}) averaged {float(row.avg_discount):.0f}% discount "
                f"across {row.txns} transactions (avg revenue/unit: {_fmt(rev_impact)}). "
                "Consider a tiered discount structure to protect margins."
            ),
            "evidence": {
                "sku": row.sku,
                "product": row.name,
                "avg_discount_pct": round(float(row.avg_discount), 1),
                "transactions": row.txns,
                "avg_unit_revenue": rev_impact,
            },
            "dedupe_key": f"pricing_discount:{row.sku}:{today.isoformat()}",
        })

    margin_risk = await db.execute(
        text("""
            SELECT p.name, p.sku, AVG(st.unit_price) AS avg_price,
                   AVG(st.discount) AS avg_discount
            FROM sales_transactions st
            JOIN products p ON p.id = st.product_id
            WHERE st.txn_date BETWEEN :s AND :e
              AND st.discount > 30
            GROUP BY p.id, p.name, p.sku
            HAVING AVG(st.discount) > 30 AND COUNT(*) >= 5
            ORDER BY AVG(st.discount) DESC
            LIMIT 3
        """),
        {"s": thirty, "e": today},
    )
    for row in margin_risk.all():
        found.append({
            "insight_type": "recommendation",
            "severity": "warning",
            "title": f"Margin erosion risk: {row.name}",
            "body": (
                f"{row.name} ({row.sku}) has an average discount of "
                f"{float(row.avg_discount):.0f}% across recent transactions. "
                "Sustained deep discounting may indicate a pricing strategy issue."
            ),
            "evidence": {
                "sku": row.sku,
                "product": row.name,
                "avg_discount_pct": round(float(row.avg_discount), 1),
            },
            "dedupe_key": f"margin_risk:{row.sku}:{today.isoformat()}",
        })

    return found


async def region_recommendations(db: AsyncSession, today: date) -> list[dict[str, Any]]:
    found = []
    thirty = today - timedelta(days=30)

    reg = await db.execute(
        text("""
            SELECT region, SUM(total_amount) AS revenue, COUNT(*) AS orders,
                   ROW_NUMBER() OVER (ORDER BY SUM(total_amount) DESC) AS rnk
            FROM sales_transactions
            WHERE txn_date BETWEEN :s AND :e
            GROUP BY region
            ORDER BY revenue DESC
        """),
        {"s": thirty, "e": today},
    )
    regions = reg.all()
    if len(regions) >= 2:
        top_region = regions[0]
        low_regions = [r for r in regions if float(r.revenue) < float(top_region.revenue) * 0.25]
        for lr in low_regions[:2]:
            gap = round((1 - lr.revenue / top_region.revenue) * 100, 1)
            found.append({
                "insight_type": "recommendation",
                "severity": "info",
                "title": f"Underperforming region: {lr.region}",
                "body": (
                    f"{lr.region} generated {_fmt(float(lr.revenue))} in 30 days — "
                    f"{gap}% behind {top_region.region} ({_fmt(float(top_region.revenue))}). "
                    "Consider regional marketing or distribution improvements."
                ),
                "evidence": {
                    "region": lr.region,
                    "revenue_30d": float(lr.revenue),
                    "orders_30d": lr.orders,
                    "top_region": top_region.region,
                    "gap_pct": gap,
                },
                "dedupe_key": f"region_gap:{lr.region}:{today.isoformat()}",
            })

    return found


async def generate_all_recommendations(db: AsyncSession) -> list[dict[str, Any]]:
    today = await _latest_data_date(db)
    if today is None:
        today = business_today()

    all_recs: list[dict[str, Any]] = []
    for generator in (
        revenue_recommendations,
        cost_recommendations,
        pricing_recommendations,
        region_recommendations,
    ):
        try:
            all_recs.extend(await generator(db, today))
        except Exception:
            import logging
            logging.getLogger(__name__).exception("recommendation %s failed", generator.__name__)

    return all_recs


async def _latest_data_date(db: AsyncSession) -> date | None:
    value = (
        await db.execute(
            text("SELECT MAX(snapshot_date) FROM kpi_snapshots WHERE metric = 'revenue'")
        )
    ).scalar_one()
    return value


# ── role scoping + impact ranking ──────────────────────────────────────────

ROLE_RANK = {"analyst": 1, "manager": 2, "admin": 3}

# Margin-pricing evidence (average discount %, margin erosion) is the sensitive
# part of recommendations; only managers and admins see those bodies verbatim.
_SENSITIVE_KINDS = {"margin_risk", "pricing_discount"}


def _impact_estimate(rec: dict[str, Any]) -> dict[str, Any]:
    """Attach a rough monetary impact estimate from whatever evidence exists."""
    ev = rec.get("evidence") or {}
    estimate: float | None = None
    basis: str | None = None

    if "gap_pct" in ev and "revenue_30d" in ev:
        estimate = float(ev["revenue_30d"]) * (float(ev["gap_pct"]) / 100.0)
        basis = "30d revenue gap"
    elif "peak_revenue" in ev and "avg_daily_revenue_90d" in ev:
        estimate = float(ev["peak_revenue"]) - float(ev["avg_daily_revenue_90d"])
        basis = "peak vs average daily revenue"
    elif "amount_30d" in ev:
        estimate = float(ev["amount_30d"])
        basis = "30-day category spend"
    elif "revenue_30d" in ev:
        estimate = float(ev["revenue_30d"])
        basis = "30-day revenue"

    if estimate is None:
        return {}
    return {"impact_estimate": round(estimate, 2), "impact_basis": basis}


async def scope_recommendations(
    db: AsyncSession,
    recs: list[dict[str, Any]],
    user: Any,
) -> list[dict[str, Any]]:
    """Role-scope and impact-rank recommendations for one user.

    Sensitive pricing/margin recommendations are trimmed to a short title for
    analysts; managers and admins see full bodies. Results are sorted by
    estimated monetary impact descending so the assistant leads with the
    highest-value suggestion.
    """
    role = getattr(user, "role", "analyst")
    access_level = ROLE_RANK.get(role, ROLE_RANK["analyst"])

    scored: list[tuple[float, dict[str, Any]]] = []
    for rec in recs:
        if not isinstance(rec, dict) or rec.get("insight_type") != "recommendation":
            continue
        kind = rec.get("dedupe_key", "").split(":")[0]
        if kind in _SENSITIVE_KINDS and access_level < ROLE_RANK["manager"]:
            # analyst sees the headline only, not margins/discount details
            rec = dict(rec)
            rec["body"] = "Details available to managers and admins."
            keep = {k: v for k, v in rec.get("evidence", {}).items() if k in ("sku", "product")}
            rec["evidence"] = keep
        impact = _impact_estimate(rec)
        rec = {**rec, **impact}
        estimate = float(impact.get("impact_estimate") or 0)
        scored.append((estimate, rec))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [rec for _, rec in scored]
