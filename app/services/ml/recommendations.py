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
            found.append(
                {
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
                }
            )

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
        avg_val = float(avg.scalar_one() or 0)
        if avg_val > 0 and float(peak_row.revenue) > avg_val * 2.5:
            found.append(
                {
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
                }
            )

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
                found.append(
                    {
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
                    }
                )

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
                found.append(
                    {
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
                    }
                )

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
        found.append(
            {
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
            }
        )

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
        found.append(
            {
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
            }
        )

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
            gap = round((1 - float(lr.revenue) / float(top_region.revenue)) * 100, 1)
            found.append(
                {
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
                        "orders_30d": int(lr.orders),
                        "top_region": top_region.region,
                        "gap_pct": gap,
                    },
                    "dedupe_key": f"region_gap:{lr.region}:{today.isoformat()}",
                }
            )

    return found


async def diagnostic_recommendations(db: AsyncSession, today: date) -> list[dict[str, Any]]:
    """Suggestions derived from *why* the numbers moved, not just what they are.

    The other generators fire on levels and gaps ("this channel is behind that
    one"). These fire on movement and structure: the single product responsible
    for a decline, an account that stopped buying outright, a month heading for
    a miss, and revenue resting on too few names. Each one names the cause, so
    the suggested action has somewhere specific to land.
    """
    from app.services.analytics.queries import Filters, kpi_summary
    from app.services.ml.diagnostics import (
        analyse_concentration,
        explain_change,
        price_volume_bridge,
    )
    from app.services.ml.projections import month_bounds, project_current_period

    found: list[dict[str, Any]] = []
    window_days = 30
    current = (today - timedelta(days=window_days - 1), today)
    previous = (
        current[0] - timedelta(days=window_days),
        current[0] - timedelta(days=1),
    )

    # ── the product behind a decline ──────────────────────────────────────
    breakdown = await explain_change(db, "product", current, previous, top_n=3)
    if breakdown.total_delta < 0 and breakdown.drags:
        worst = breakdown.drags[0]
        share = abs(worst.contribution_pct)
        found.append(
            {
                "insight_type": "recommendation",
                "severity": "warning" if share >= 40 else "info",
                "title": f"{worst.key} is the biggest drag on revenue",
                "body": (
                    f"Revenue fell {_fmt(abs(breakdown.total_delta))} over the last "
                    f"{window_days} days, and {worst.key} accounts for "
                    f"{_fmt(abs(worst.delta))} of that — {share:.0f}% of the total movement "
                    f"({_fmt(worst.previous)} → {_fmt(worst.current)}). Fixing this one line "
                    "recovers more than any broad campaign."
                ),
                "evidence": {
                    "product": worst.key,
                    "revenue_current": worst.current,
                    "revenue_previous": worst.previous,
                    "revenue_delta": worst.delta,
                    "contribution_pct": worst.contribution_pct,
                    "period_days": window_days,
                },
                "dedupe_key": f"drag_product:{worst.key}:{today.isoformat()}",
            }
        )

    # A member that went to exactly zero is a lost account, not soft demand —
    # a completely different conversation, so it gets its own suggestion.
    for lost in breakdown.lost_members[:2]:
        found.append(
            {
                "insight_type": "recommendation",
                "severity": "warning",
                "title": f"{lost} stopped selling entirely",
                "body": (
                    f"{lost} sold in the previous {window_days} days and has sold nothing "
                    "since. A clean drop to zero usually means a lost account or a stockout "
                    "rather than falling demand — worth a call before it is treated as a "
                    "trend."
                ),
                "evidence": {"product": lost, "period_days": window_days},
                "dedupe_key": f"lost_product:{lost}:{today.isoformat()}",
            }
        )

    # ── volume problem or value problem ───────────────────────────────────
    cards = {
        c["metric"]: c
        for c in await kpi_summary(db, Filters(date_from=current[0], date_to=current[1]))
    }
    rev, orders = cards.get("revenue"), cards.get("orders")
    if rev and orders and rev.get("previous_value") and orders.get("previous_value"):
        bridge = price_volume_bridge(
            orders_current=float(orders["value"] or 0),
            orders_previous=float(orders["previous_value"] or 0),
            revenue_current=float(rev["value"] or 0),
            revenue_previous=float(rev["previous_value"] or 0),
        )
        if bridge.revenue_delta < 0 and bridge.verdict != "no material change":
            if abs(bridge.volume_effect) > abs(bridge.value_effect):
                cause, action = (
                    "fewer orders",
                    "acquisition and reactivation move this; discounting will not",
                )
            else:
                cause, action = (
                    "smaller orders",
                    "bundling and minimum-order incentives move this; more traffic will not",
                )
            found.append(
                {
                    "insight_type": "recommendation",
                    "severity": "info",
                    "title": f"Revenue decline is a {cause} problem",
                    "body": (
                        f"Revenue is down {_fmt(abs(bridge.revenue_delta))} over "
                        f"{window_days} days. Order volume accounts for "
                        f"{_fmt(bridge.volume_effect)} and order value for "
                        f"{_fmt(bridge.value_effect)}, so this is {cause} — {action}."
                    ),
                    "evidence": {
                        "revenue_delta": bridge.revenue_delta,
                        "volume_effect": bridge.volume_effect,
                        "value_effect": bridge.value_effect,
                        "orders_current": bridge.orders_current,
                        "orders_previous": bridge.orders_previous,
                        "aov_current": bridge.aov_current,
                        "aov_previous": bridge.aov_previous,
                    },
                    "dedupe_key": f"revenue_bridge:{bridge.verdict}:{today.isoformat()}",
                }
            )

    # ── the month is heading for a miss ───────────────────────────────────
    projection = await project_current_period(db, metric="revenue", period="month")
    month_start, _ = month_bounds(today)
    if projection.days_elapsed >= 5 and projection.days_remaining >= 3:
        prev_month_end = month_start - timedelta(days=1)
        prev_cards = {
            c["metric"]: c
            for c in await kpi_summary(
                db,
                Filters(date_from=prev_month_end.replace(day=1), date_to=prev_month_end),
            )
        }
        last_month = float((prev_cards.get("revenue") or {}).get("value") or 0.0)
        if last_month and projection.projected_total < last_month * 0.95:
            shortfall = last_month - projection.projected_total
            found.append(
                {
                    "insight_type": "recommendation",
                    "severity": "warning",
                    "title": f"{projection.period_label} is tracking below last month",
                    "body": (
                        f"At the current run rate of {_fmt(projection.daily_run_rate)}/day, "
                        f"{projection.period_label} lands near "
                        f"{_fmt(projection.projected_total)} against "
                        f"{_fmt(last_month)} last month — a shortfall of "
                        f"{_fmt(shortfall)} with {projection.days_remaining} day(s) left to "
                        f"act. Closing it needs about "
                        f"{_fmt(shortfall / projection.days_remaining)} extra per day."
                    ),
                    "evidence": {
                        "projected_total": projection.projected_total,
                        "lower_bound": projection.lower_bound,
                        "upper_bound": projection.upper_bound,
                        "last_month_revenue": last_month,
                        "shortfall": shortfall,
                        "days_remaining": projection.days_remaining,
                        "method": projection.method,
                    },
                    "dedupe_key": f"period_shortfall:{projection.period_label}:{today.isoformat()}",
                }
            )

    # ── too much resting on too few ───────────────────────────────────────
    conc = await analyse_concentration(db, "product", *current)
    if conc.members >= 3 and conc.risk.startswith("high"):
        top1 = round(conc.top1_share_pct, 1)
        top3 = round(conc.top3_share_pct, 1)
        hhi = round(conc.hhi, 3)
        found.append(
            {
                "insight_type": "recommendation",
                "severity": "info",
                "title": "Revenue is concentrated in very few products",
                "body": (
                    f"{conc.leaders[0]} alone is {top1}% of revenue and the "
                    f"top three are {top3}% across {conc.members} products "
                    f"(HHI {hhi}). Losing one of them would move the headline number on "
                    "its own — worth knowing before it happens rather than after."
                ),
                "evidence": {
                    "top1_share_pct": top1,
                    "top3_share_pct": top3,
                    "hhi": hhi,
                    "members": conc.members,
                    "leaders": conc.leaders,
                },
                "dedupe_key": f"concentration:product:{today.isoformat()}",
            }
        )

    return found


async def persist_recommendations(db: AsyncSession) -> dict[str, int]:
    """Generate recommendations and store new ones as ``insights`` rows.

    Shared by the manual "Generate now" endpoint and the nightly scheduler job
    so both paths do exactly the same work — dedupe via ``dedupe_key`` means
    running this twice on the same day is a no-op the second time. New
    warning-severity recommendations also drop an in-app notification for
    admins/managers, the same way alert rules do, so a scheduled run is
    visible without anyone having to revisit this page.
    """
    from sqlalchemy import select
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from app.models import Insight, Notification, Profile

    today = business_today()
    recs = await generate_all_recommendations(db)
    created = 0
    new_warnings: list[dict[str, Any]] = []
    for r in recs:
        # impact_basis is an explanation aid, not a stored column
        payload = {k: v for k, v in r.items() if k != "impact_basis"}
        payload.update(period_start=today, period_end=today)
        stmt = (
            pg_insert(Insight)
            .values(**payload)
            .on_conflict_do_nothing(index_elements=["dedupe_key"])
            .returning(Insight.id)
        )
        result = await db.execute(stmt)
        if result.first() is not None:
            created += 1
            if r.get("severity") == "warning":
                new_warnings.append(r)

    if new_warnings:
        recipients = (
            (
                await db.execute(
                    select(Profile).where(
                        Profile.role.in_(["admin", "manager"]), Profile.is_active.is_(True)
                    )
                )
            )
            .scalars()
            .all()
        )
        title = (
            f"{len(new_warnings)} new recommendation(s) need attention"
            if len(new_warnings) > 1
            else new_warnings[0]["title"]
        )
        body = (
            new_warnings[0]["body"]
            if len(new_warnings) == 1
            else "; ".join(w["title"] for w in new_warnings)
        )
        for user in recipients:
            db.add(Notification(user_id=user.id, title=title, body=body))

    await db.commit()
    return {"generated": len(recs), "new": created}


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
        diagnostic_recommendations,
    ):
        try:
            all_recs.extend(await generator(db, today))
        except Exception:
            import logging

            logging.getLogger(__name__).exception("recommendation %s failed", generator.__name__)

    # Phase 8: every recommendation leaves with WHY (evidence) / EXPECTED
    # IMPACT / PRIORITY / CONFIDENCE / ACTION attached, so neither the live
    # list nor the persisted insights ever present a bare "do this".
    enriched: list[dict[str, Any]] = []
    for rec in all_recs:
        impact = _impact_estimate(rec)
        rec = {**rec, **impact}
        rec["priority"] = _priority_for(rec)
        rec["action"] = _default_action_for(rec)
        enriched.append(rec)
    return enriched


def _priority_for(rec: dict[str, Any]) -> str:
    """Priority = impact × severity (components shown in evidence/priority_basis)."""
    estimate = float(rec.get("impact_estimate") or 0)
    severity = rec.get("severity", "info")
    if severity == "critical" or estimate >= 200_000:
        return "high"
    if severity == "warning" or estimate >= 50_000:
        return "medium"
    return "low"


def _default_action_for(rec: dict[str, Any]) -> str:
    title = str(rec.get("title", "")).lower()
    if any(w in title for w in ("stock", "reorder", "inventory", "stockout")):
        return "Place a reorder / review the reorder level for the affected SKUs"
    if any(w in title for w in ("discount", "price", "margin", "pricing")):
        return "Review unit pricing / discount levels for the affected products"
    if any(w in title for w in ("expense", "cost", "spend", "overhead")):
        return "Investigate the cost line; negotiate or cut where evidence allows"
    if any(w in title for w in ("region", "district", "Biratnagar", "Pokhara")):
        return "Investigate the region's performance gap and plan a local push"
    return "Review the reported variance and its underlying drivers"


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
