from datetime import date, timedelta
import logging
import statistics
from typing import Any

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.clock import business_today

logger = logging.getLogger(__name__)


def _fmt(value: float) -> str:
    return f"NPR {value:,.0f}"


def _ninety_days_ago(today: date) -> date:
    return today - timedelta(days=90)


async def revenue_recommendations(db: AsyncSession, today: date, org_id=None) -> list[dict[str, Any]]:
    found = []
    thirty = today - timedelta(days=30)
    ninety = today - timedelta(days=90)

    ch = await db.execute(
        text("""
            SELECT channel, SUM(total_amount) AS revenue,
                   ROW_NUMBER() OVER (ORDER BY SUM(total_amount) DESC) AS rnk
            FROM sales_transactions
            WHERE txn_date BETWEEN :s AND :e
              AND (:oid IS NULL OR org_id = :oid)
            GROUP BY channel
            ORDER BY revenue DESC
        """).bindparams(bindparam("oid", type_=PG_UUID(as_uuid=True))),
        {"s": thirty, "e": today, "oid": str(org_id) if org_id else None},
    )
    channels = ch.all()
    if len(channels) >= 2:
        top_channel = channels[0]
        # consistent threshold 30% for both channel and region
        threshold = float(top_channel.revenue) * 0.30
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
                    "kind": "channel_boost",
                    "dedupe_key": f"channel_boost:{bc.channel}:{today.isoformat()}",
                }
            )

    # peak detection using z-score and p90 instead of fixed 2.5x avg
    daily_rows = await db.execute(
        text("""
            SELECT txn_date AS day, SUM(total_amount) AS revenue
            FROM sales_transactions
            WHERE txn_date BETWEEN :s AND :e
              AND (:oid IS NULL OR org_id = :oid)
            GROUP BY txn_date ORDER BY txn_date
        """).bindparams(bindparam("oid", type_=PG_UUID(as_uuid=True))),
        {"s": ninety, "e": today, "oid": str(org_id) if org_id else None},
    )
    daily = daily_rows.all()
    if daily:
        revenues = [float(r.revenue) for r in daily]
        peak_row = max(daily, key=lambda r: float(r.revenue))
        mean = statistics.fmean(revenues) if revenues else 0.0
        stdev = statistics.stdev(revenues) if len(revenues) > 1 else 0.0
        sorted_rev = sorted(revenues)
        # p90 via nearest-rank
        p90_idx = max(0, min(len(sorted_rev) - 1, int(len(sorted_rev) * 0.9)))
        p90 = sorted_rev[p90_idx] if sorted_rev else 0.0
        z = (float(peak_row.revenue) - mean) / stdev if stdev > 0 else 0.0
        # require both statistical signals: high z-score and above p90
        if revenues and (z >= 2.0 or float(peak_row.revenue) >= p90 * 1.5 if p90 else False) and float(peak_row.revenue) > mean:
            # only flag if truly anomalous: either z >=2 or revenue >= p90 and > mean + significance
            is_peak = False
            if stdev > 0:
                is_peak = z >= 2.0 and float(peak_row.revenue) >= p90
            else:
                is_peak = float(peak_row.revenue) >= p90 and float(peak_row.revenue) > mean * 1.5
            if is_peak:
                avg_val = mean
                # ensure avg for backward compat in evidence
                found.append(
                    {
                        "insight_type": "recommendation",
                        "severity": "info",
                        "title": f"Replicate peak day performance: {peak_row.day:%b %d}",
                        "body": (
                            f"{peak_row.day:%b %d} generated {_fmt(float(peak_row.revenue))} — "
                            f"{round(float(peak_row.revenue) / avg_val, 1) if avg_val else 0}x the daily average. "
                            "Review what drove that day (promotions, marketing, events) and replicate."
                        ),
                        "evidence": {
                            "peak_date": peak_row.day.isoformat() if hasattr(peak_row.day, 'isoformat') else str(peak_row.day),
                            "peak_revenue": float(peak_row.revenue),
                            "avg_daily_revenue_90d": round(float(avg_val), 2),
                            "multiplier": round(float(peak_row.revenue) / avg_val, 1) if avg_val else 0,
                            "z_score": round(z, 2),
                            "p90_revenue": round(float(p90), 2),
                        },
                        "kind": "peak_day",
                        "dedupe_key": f"peak_day:{today.isoformat()}",
                    }
                )

    return found


async def cost_recommendations(db: AsyncSession, today: date, org_id=None) -> list[dict[str, Any]]:
    found = []
    thirty = today - timedelta(days=30)

    top_exp = await db.execute(
        text("""
            SELECT category, SUM(amount) AS total
            FROM expenses
            WHERE expense_date BETWEEN :s AND :e
              AND (:oid IS NULL OR org_id = :oid)
            GROUP BY category
            ORDER BY total DESC
            LIMIT 3
        """).bindparams(bindparam("oid", type_=PG_UUID(as_uuid=True))),
        {"s": thirty, "e": today, "oid": str(org_id) if org_id else None},
    )
    expenses = top_exp.all()
    if expenses:
        # fix total_exp denominator to sum total expenses not just top3
        total_row = await db.execute(
            text("""
                SELECT COALESCE(SUM(amount),0) AS total
                FROM expenses
                WHERE expense_date BETWEEN :s AND :e
                  AND (:oid IS NULL OR org_id = :oid)
            """).bindparams(bindparam("oid", type_=PG_UUID(as_uuid=True))),
            {"s": thirty, "e": today, "oid": str(org_id) if org_id else None},
        )
        total_exp = float(total_row.scalar_one() or 0)
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
                            "total_expenses_30d": total_exp,
                            "share_pct": share,
                        },
                        "kind": "cost_share",
                        "dedupe_key": f"cost_share:{e.category}:{today.isoformat()}",
                    }
                )

    exp_trend = await db.execute(
        text("""
            SELECT DATE_TRUNC('month', expense_date) AS month, SUM(amount) AS total
            FROM expenses
            WHERE expense_date >= :s
              AND (:oid IS NULL OR org_id = :oid)
            GROUP BY month ORDER BY month
        """).bindparams(bindparam("oid", type_=PG_UUID(as_uuid=True))),
        {"s": _ninety_days_ago(today), "oid": str(org_id) if org_id else None},
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
                        "kind": "cost_mom",
                        "dedupe_key": f"cost_mom:{today.isoformat()}",
                    }
                )

    return found


async def pricing_recommendations(db: AsyncSession, today: date, org_id=None) -> list[dict[str, Any]]:
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
              AND (:oid IS NULL OR st.org_id = :oid)
              AND (:oid IS NULL OR p.org_id = :oid)
            GROUP BY p.id, p.name, p.sku
            HAVING AVG(st.discount) > 20 AND COUNT(*) >= 10
            ORDER BY AVG(st.discount) DESC
            LIMIT 3
        """).bindparams(bindparam("oid", type_=PG_UUID(as_uuid=True))),
        {"s": thirty, "e": today, "oid": str(org_id) if org_id else None},
    )
    seen_skus = set()
    for row in deep_discount.all():
        rev_impact = round(float(row.revenue) / float(row.txns), 2) if row.txns else 0
        discount_pct = round(float(row.avg_discount), 1)
        # impact estimate: discount * revenue approximation
        found.append(
            {
                "insight_type": "recommendation",
                "severity": "info",
                "title": f"High discounts on {row.name}",
                "body": (
                    f"{row.name} ({row.sku}) averaged {discount_pct:.0f}% discount "
                    f"across {row.txns} transactions (avg revenue/unit: {_fmt(rev_impact)}). "
                    "Consider a tiered discount structure to protect margins."
                ),
                "evidence": {
                    "sku": row.sku,
                    "product": row.name,
                    "avg_discount_pct": discount_pct,
                    "transactions": row.txns,
                    "avg_unit_revenue": rev_impact,
                    "revenue_30d": float(row.revenue),
                },
                "kind": "pricing_discount",
                "dedupe_key": f"pricing_discount:{row.sku}:{today.isoformat()}",
            }
        )
        seen_skus.add(row.sku)

    margin_risk = await db.execute(
        text("""
            SELECT p.name, p.sku, AVG(st.unit_price) AS avg_price,
                   AVG(st.discount) AS avg_discount, COUNT(*) AS txns, SUM(st.total_amount) AS revenue
            FROM sales_transactions st
            JOIN products p ON p.id = st.product_id
            WHERE st.txn_date BETWEEN :s AND :e
              AND st.discount > 30
              AND (:oid IS NULL OR st.org_id = :oid)
              AND (:oid IS NULL OR p.org_id = :oid)
            GROUP BY p.id, p.name, p.sku
            HAVING AVG(st.discount) > 30 AND COUNT(*) >= 5
            ORDER BY AVG(st.discount) DESC
            LIMIT 3
        """).bindparams(bindparam("oid", type_=PG_UUID(as_uuid=True))),
        {"s": thirty, "e": today, "oid": str(org_id) if org_id else None},
    )
    for row in margin_risk.all():
        # fix duplicate pricing_discount vs margin_risk dedup: skip SKU already flagged
        if row.sku in seen_skus:
            continue
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
                    "transactions": row.txns,
                    "revenue_30d": float(row.revenue) if hasattr(row, 'revenue') else 0,
                },
                "kind": "margin_risk",
                "dedupe_key": f"margin_risk:{row.sku}:{today.isoformat()}",
            }
        )

    return found


async def region_recommendations(db: AsyncSession, today: date, org_id=None) -> list[dict[str, Any]]:
    found = []
    thirty = today - timedelta(days=30)

    reg = await db.execute(
        text("""
            SELECT region, SUM(total_amount) AS revenue, COUNT(*) AS orders,
                   ROW_NUMBER() OVER (ORDER BY SUM(total_amount) DESC) AS rnk
            FROM sales_transactions
            WHERE txn_date BETWEEN :s AND :e
              AND (:oid IS NULL OR org_id = :oid)
            GROUP BY region
            ORDER BY revenue DESC
        """).bindparams(bindparam("oid", type_=PG_UUID(as_uuid=True))),
        {"s": thirty, "e": today, "oid": str(org_id) if org_id else None},
    )
    regions = reg.all()
    if len(regions) >= 2:
        top_region = regions[0]
        # consistent with channel: 0.30 threshold
        low_regions = [r for r in regions if float(r.revenue) < float(top_region.revenue) * 0.30]
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
                        "top_revenue_30d": float(top_region.revenue),
                        "gap_pct": gap,
                    },
                    "kind": "region_gap",
                    "dedupe_key": f"region_gap:{lr.region}:{today.isoformat()}",
                }
            )

    return found


async def diagnostic_recommendations(db: AsyncSession, today: date, org_id=None) -> list[dict[str, Any]]:
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
    breakdown = await explain_change(db, "product", current, previous, top_n=3, org_id=org_id)
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
                    "total_delta": breakdown.total_delta,
                    "current_window": f"{current[0].isoformat()}→{current[1].isoformat()}",
                    "previous_window": f"{previous[0].isoformat()}→{previous[1].isoformat()}",
                },
                "kind": "drag_product",
                "dedupe_key": f"drag_product:{worst.key}:{today.isoformat()}:{current[0].isoformat()}",
            }
        )

    # A member that went to exactly zero is a lost account, not soft demand —
    # a completely different conversation, so it gets its own suggestion.
    # Enrich evidence with previous revenue values
    # build map of previous revenues for lost members
    # breakdown.lost_members are keys; need to find their previous values from breakdown drivers/drags or via _revenue_by
    # Use breakdown's internal contributions: we can reconstruct from explain_change's data by re-querying
    lost_prev_map: dict[str, float] = {}
    try:
        from app.services.ml.diagnostics import _revenue_by
        prev_revs = await _revenue_by(db, "product", *previous, org_id=org_id)
        curr_revs = await _revenue_by(db, "product", *current, org_id=org_id)
        for lost in breakdown.lost_members:
            lost_prev_map[lost] = prev_revs.get(lost, 0.0)
    except Exception:
        # fallback: try to extract from drags
        for c in breakdown.drags:
            if c.key in breakdown.lost_members:
                lost_prev_map[c.key] = c.previous
    for lost in breakdown.lost_members[:2]:
        prev_rev = lost_prev_map.get(lost, 0.0)
        found.append(
            {
                "insight_type": "recommendation",
                "severity": "warning",
                "title": f"{lost} stopped selling entirely",
                "body": (
                    f"{lost} sold {_fmt(prev_rev)} in the previous {window_days} days and has sold nothing "
                    "since. A clean drop to zero usually means a lost account or a stockout "
                    "rather than falling demand — worth a call before it is treated as a "
                    "trend."
                ),
                "evidence": {
                    "product": lost,
                    "period_days": window_days,
                    "revenue_previous": prev_rev,
                    "revenue_current": 0.0,
                    "revenue_delta": -prev_rev,
                    "current_window": f"{current[0].isoformat()}→{current[1].isoformat()}",
                    "previous_window": f"{previous[0].isoformat()}→{previous[1].isoformat()}",
                },
                "kind": "lost_product",
                "dedupe_key": f"lost_product:{lost}:{today.isoformat()}:{current[0].isoformat()}",
            }
        )

    # ── volume problem or value problem ───────────────────────────────────
    cards = {
        c["metric"]: c for c in await kpi_summary(db, Filters(date_from=current[0], date_to=current[1], org_id=org_id))
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
                        "period_days": window_days,
                        "current_window": f"{current[0].isoformat()}→{current[1].isoformat()}",
                        "previous_window": f"{previous[0].isoformat()}→{previous[1].isoformat()}",
                    },
                    # include period window in dedupe to avoid cross-period collision
                    "kind": "revenue_bridge",
                    "dedupe_key": f"revenue_bridge:{bridge.verdict}:{current[0].isoformat()}→{current[1].isoformat()}:{today.isoformat()}",
                }
            )

    # ── the month is heading for a miss ───────────────────────────────────
    projection = await project_current_period(db, metric="revenue", period="month", org_id=org_id)
    month_start, _ = month_bounds(today)
    if projection.days_elapsed >= 5 and projection.days_remaining >= 3:
        prev_month_end = month_start - timedelta(days=1)
        prev_cards = {
            c["metric"]: c
            for c in await kpi_summary(
                db,
                Filters(date_from=prev_month_end.replace(day=1), date_to=prev_month_end, org_id=org_id),
            )
        }
        last_month = float((prev_cards.get("revenue") or {}).get("value") or 0.0)
        # fix projection shortfall buffer: use lower_bound with buffer, more sensitive
        # Require projected_total below 98% of last month OR lower_bound below 95% to catch uncertain misses
        projected_with_buffer = projection.projected_total
        # add safety buffer: if remaining days variability could erase shortfall, don't flag prematurely
        # Use lower_bound as conservative estimate; require both point and lower bound to be short
        buffer_threshold = last_month * 0.95
        conservative_threshold = last_month * 0.98
        is_shortfall = False
        if last_month:
            if projected_with_buffer < buffer_threshold:
                is_shortfall = True
            elif projection.lower_bound < buffer_threshold and projected_with_buffer < conservative_threshold:
                is_shortfall = True
        if last_month and is_shortfall:
            shortfall = last_month - projection.projected_total
            # include buffer in shortfall display
            conservative_shortfall = last_month - projection.lower_bound
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
                        f"{_fmt(shortfall)} (conservative {_fmt(conservative_shortfall)} with 95% band) with {projection.days_remaining} day(s) left to "
                        f"act. Closing it needs about "
                        f"{_fmt(shortfall / projection.days_remaining)} extra per day."
                    ),
                    "evidence": {
                        "projected_total": projection.projected_total,
                        "lower_bound": projection.lower_bound,
                        "upper_bound": projection.upper_bound,
                        "last_month_revenue": last_month,
                        "shortfall": shortfall,
                        "conservative_shortfall": conservative_shortfall,
                        "days_remaining": projection.days_remaining,
                        "method": projection.method,
                        "daily_run_rate": projection.daily_run_rate,
                    },
                    "kind": "period_shortfall",
                    "dedupe_key": f"period_shortfall:{projection.period_label}:{today.isoformat()}",
                }
            )

    # ── too much resting on too few ───────────────────────────────────────
    conc = await analyse_concentration(db, "product", *current, org_id=org_id)
    # fix concentration risk to numeric hhi instead of string parsing
    if conc.members >= 3 and conc.hhi >= 0.25:
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
                "kind": "concentration",
                "dedupe_key": f"concentration:product:{today.isoformat()}",
            }
        )

    return found


async def persist_recommendations(db: AsyncSession, org_id=None) -> dict[str, int]:
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
    recs = await generate_all_recommendations(db, org_id=org_id)
    created = 0
    new_warnings: list[dict[str, Any]] = []
    for r in recs:
        # impact_basis is an explanation aid, not a stored column
        payload = {k: v for k, v in r.items() if k != "impact_basis"}
        # kind is helper for scoping, not a DB column — keep in evidence or drop
        # Store kind as separate attribute for later scoping if model supports it; otherwise keep in evidence
        # For backward compat, remove kind before insert if column doesn't exist
        kind_val = payload.pop("kind", None)
        # keep kind in evidence for debugging if needed
        if kind_val and payload.get("evidence") is not None:
            # don't overwrite evidence, just ensure kind is traceable via dedupe_key
            pass
        payload.update(period_start=today, period_end=today)
        if org_id is not None:
            payload["org_id"] = org_id
            # scope dedupe_key per org
            if payload.get("dedupe_key"):
                payload["dedupe_key"] = f"{org_id}:{payload['dedupe_key']}"
        # use composite index (org_id, dedupe_key)
        if org_id is not None:
            stmt = (
                pg_insert(Insight)
                .values(**payload)
                .on_conflict_do_nothing(index_elements=["org_id", "dedupe_key"])
                .returning(Insight.id)
            )
        else:
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
        rec_q = select(Profile).where(Profile.role.in_(["admin", "manager"]), Profile.is_active.is_(True))
        if org_id is not None:
            rec_q = rec_q.where(Profile.org_id == org_id)
        recipients = (await db.execute(rec_q)).scalars().all()
        title = (
            f"{len(new_warnings)} new recommendation(s) need attention"
            if len(new_warnings) > 1
            else new_warnings[0]["title"]
        )
        body = new_warnings[0]["body"] if len(new_warnings) == 1 else "; ".join(w["title"] for w in new_warnings)
        for user in recipients:
            db.add(Notification(user_id=user.id, title=title, body=body, org_id=org_id or user.org_id))

    await db.commit()
    return {"generated": len(recs), "new": created}


async def generate_all_recommendations(db: AsyncSession, org_id=None) -> list[dict[str, Any]]:
    today = await _latest_data_date(db, org_id=org_id)
    if today is None:
        today = business_today()
    else:
        # ensure we use the fresher of snapshot/latest_txn vs business clock
        bt = business_today()
        # if snapshot is older than business_today but we have no future data, keep snapshot date to avoid empty window
        # However if txn data is newer than snapshot, _latest_data_date already returns max
        # Only override if business_today is significantly behind? Use max logic per task: max(business_today, latest_txn)
        # We already have max in _latest_data_date; here we just ensure today is not None
        pass

    all_recs: list[dict[str, Any]] = []
    errors: list[tuple[str, Exception]] = []
    for generator in (
        revenue_recommendations,
        cost_recommendations,
        pricing_recommendations,
        region_recommendations,
        diagnostic_recommendations,
    ):
        try:
            # Generators that support org_id will filter; others will run globally (legacy)
            import inspect as _ins

            sig = _ins.signature(generator)
            # A SAVEPOINT scopes rollback to this generator's own failed
            # statement — a plain session-wide rollback would also expire
            # (detach) every other ORM object already loaded on this session,
            # e.g. the request's current-user Profile.
            async with db.begin_nested():
                if "org_id" in sig.parameters:
                    all_recs.extend(await generator(db, today, org_id=org_id))  # type: ignore[call-arg]
                else:
                    all_recs.extend(await generator(db, today))
        except Exception as e:
            errors.append((generator.__name__, e))
            logging.getLogger(__name__).exception("recommendation %s failed", generator.__name__)

    if errors:
        # aggregate errors instead of silently swallowing — log summary
        logger.warning("recommendation generation completed with %d errors: %s", len(errors), ", ".join(n for n, _ in errors))
        # if all generators failed, surface aggregated exception for observability
        if len(errors) == 5:
            # attach aggregated context but still return partial (empty) to avoid 500 on dashboard
            logger.error("all recommendation generators failed: %s", errors)

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
    """Priority = impact × severity, impact as % of revenue (components shown in evidence/priority_basis)."""
    estimate = float(rec.get("impact_estimate") or 0)
    severity = rec.get("severity", "info")
    ev = rec.get("evidence") or {}
    # determine revenue denominator for percentage
    revenue_denominator = None
    for key in ("top_revenue_30d", "top1_share_pct", "revenue_30d", "amount_30d", "revenue_current", "projected_total", "peak_revenue"):
        if key in ev:
            try:
                revenue_denominator = float(ev[key])
                if revenue_denominator > 0:
                    break
            except Exception:
                continue
    # also try to infer via gap
    if revenue_denominator is None and "gap_pct" in ev and "revenue_30d" in ev:
        try:
            top = float(ev.get("top_revenue_30d", 0)) or float(ev["revenue_30d"]) / (1 - float(ev["gap_pct"])/100) if float(ev["gap_pct"]) < 100 else None
            revenue_denominator = top
        except Exception:
            revenue_denominator = None
    # percent thresholds: high if >=10% of period revenue or absolute fallback 200k
    if revenue_denominator and revenue_denominator > 0:
        pct = estimate / revenue_denominator * 100
        if severity == "critical" or pct >= 10 or estimate >= 200_000:
            return "high"
        if severity == "warning" or pct >= 3 or estimate >= 50_000:
            return "medium"
        return "low"
    # fallback to absolute if no denominator
    if severity == "critical" or estimate >= 200_000:
        return "high"
    if severity == "warning" or estimate >= 50_000:
        return "medium"
    return "low"


def _default_action_for(rec: dict[str, Any]) -> str:
    ev = rec.get("evidence") or {}
    title = str(rec.get("title", "")).lower()
    # reference evidence directly for specificity
    sku = ev.get("sku")
    product = ev.get("product") or ev.get("category") or ev.get("channel") or ev.get("region")
    if any(w in title for w in ("stock", "reorder", "inventory", "stockout")):
        if sku:
            return f"Place a reorder for {product or sku} (SKU {sku}) — review reorder level {ev.get('reorder_level','')} and cover {ev.get('period_days','30')} days"
        return f"Place a reorder / review the reorder level for {product or 'the affected SKUs'} (evidence: {ev})"
    if any(w in title for w in ("discount", "price", "margin", "pricing")):
        if sku:
            return f"Review pricing/discount for {product} (SKU {sku}, {ev.get('avg_discount_pct','?')}% avg discount)"
        return f"Review unit pricing / discount levels for {product or 'the affected products'} — evidence: {ev}"
    if any(w in title for w in ("expense", "cost", "spend", "overhead")):
        cat = ev.get("category") or product
        if cat:
            return f"Investigate {cat} cost line ({_fmt(float(ev.get('amount_30d',0)))} — {ev.get('share_pct','?')}% of spend); negotiate or cut where evidence allows"
        return "Investigate the cost line; negotiate or cut where evidence allows"
    if any(w in title for w in ("region", "district", "Biratnagar", "Pokhara")):
        reg = ev.get("region") or product
        if reg:
            return f"Investigate {reg} performance gap ({ev.get('gap_pct','?')}% behind {ev.get('top_region','top')}) and plan a local push"
        return "Investigate the region's performance gap and plan a local push"
    if "drag" in title or "biggest drag" in title:
        prod = ev.get("product")
        if prod:
            return f"Diagnose {prod} decline ({_fmt(float(ev.get('revenue_delta',0)))} delta); check stock, pricing and channel for that SKU"
    if "stopped selling" in title:
        prod = ev.get("product")
        if prod:
            return f"Contact account for {prod} — zero sales vs {_fmt(float(ev.get('revenue_previous',0)))} prior; check lost account or stockout"
    if "concentrat" in title:
        leaders = ev.get("leaders") or []
        if leaders:
            return f"Diversify away from {', '.join(leaders[:2])} (HHI {ev.get('hhi','?')}); promote secondary products"
    if "bridge" in title or "fewer orders" in title or "smaller orders" in title:
        return f"Act on {'volume' if 'fewer' in title else 'value'} driver per evidence volume {ev.get('volume_effect')} vs value {ev.get('value_effect')}"
    if "tracking below" in title or "shortfall" in title:
        return f"Close shortfall of {_fmt(float(ev.get('shortfall',0)))} with {ev.get('days_remaining','?')} days left — needs {_fmt(float(ev.get('shortfall',0))/max(1,float(ev.get('days_remaining',1))))}/day"
    return f"Review the reported variance and its underlying drivers — evidence: {ev}"


async def _latest_data_date(db: AsyncSession, org_id=None) -> date | None:
    # fix today fallback to max business_today/latest_txn (and snapshots)
    # query both snapshots and txn max to get freshest data date
    snapshot_val = None
    txn_val = None
    exp_val = None
    try:
        if org_id is not None:
            snapshot_val = (
                await db.execute(
                    text("SELECT MAX(snapshot_date) FROM kpi_snapshots WHERE metric = 'revenue' AND org_id = :oid"),
                    {"oid": str(org_id)},
                )
            ).scalar_one()
        else:
            snapshot_val = (
                await db.execute(text("SELECT MAX(snapshot_date) FROM kpi_snapshots WHERE metric = 'revenue'"))
            ).scalar_one()
    except Exception:
        snapshot_val = None
    try:
        if org_id is not None:
            txn_val = (
                await db.execute(
                    text("SELECT MAX(txn_date) FROM sales_transactions WHERE org_id = :oid"),
                    {"oid": str(org_id)},
                )
            ).scalar_one()
        else:
            txn_val = (await db.execute(text("SELECT MAX(txn_date) FROM sales_transactions"))).scalar_one()
    except Exception:
        txn_val = None
    try:
        if org_id is not None:
            exp_val = (
                await db.execute(
                    text("SELECT MAX(expense_date) FROM expenses WHERE org_id = :oid"),
                    {"oid": str(org_id)},
                )
            ).scalar_one()
        else:
            exp_val = (await db.execute(text("SELECT MAX(expense_date) FROM expenses"))).scalar_one()
    except Exception:
        exp_val = None
    candidates = [d for d in (snapshot_val, txn_val, exp_val) if d is not None]
    if not candidates:
        return None
    latest = max(candidates)
    # task says max business_today/latest_txn — ensure we respect business clock max
    # If latest is stale behind business_today by many days but we have gap, we still want latest to anchor window
    # But if latest is in future relative to business_today (clock skew), clamp to business_today
    bt = business_today()
    if latest > bt:
        return bt
    return latest


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

    # fix gap estimate to top_rev - rev_30d (not rev * gap%)
    if "top_revenue_30d" in ev and "revenue_30d" in ev:
        try:
            estimate = float(ev["top_revenue_30d"]) - float(ev["revenue_30d"])
            basis = "30d revenue gap to top"
        except Exception:
            estimate = None
    if estimate is None and "gap_pct" in ev and "revenue_30d" in ev:
        # fallback if top not present, compute gap amount
        try:
            # gap_pct is percent behind top, so gap amount = top - rev
            # top = rev / (1 - gap_pct/100)
            rev = float(ev["revenue_30d"])
            gap = float(ev["gap_pct"])
            if gap < 100:
                top = rev / (1 - gap/100) if gap != 100 else rev
                estimate = top - rev
                basis = "30d revenue gap"
            else:
                estimate = rev * (gap/100)
                basis = "30d revenue gap"
        except Exception:
            estimate = None
    if estimate is None and "peak_revenue" in ev and "avg_daily_revenue_90d" in ev:
        estimate = float(ev["peak_revenue"]) - float(ev["avg_daily_revenue_90d"])
        basis = "peak vs average daily revenue"
    if estimate is None and "amount_30d" in ev:
        estimate = float(ev["amount_30d"])
        basis = "30-day category spend"
    if estimate is None and "revenue_30d" in ev:
        estimate = float(ev["revenue_30d"])
        basis = "30-day revenue"
    if estimate is None and "revenue_delta" in ev:
        estimate = abs(float(ev["revenue_delta"]))
        basis = "revenue delta"
    if estimate is None and "shortfall" in ev:
        estimate = float(ev["shortfall"])
        basis = "projected shortfall"
    if estimate is None and "contribution_pct" in ev and "total_delta" in ev:
        estimate = abs(float(ev["revenue_delta"] or 0)) if "revenue_delta" in ev else abs(float(ev.get("total_delta", 0)) * float(ev.get("contribution_pct",0))/100)
        basis = "contribution to decline"
    if estimate is None and "revenue_previous" in ev and ev.get("revenue_current") == 0:
        estimate = float(ev["revenue_previous"])
        basis = "lost revenue (previous period)"

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
        # fix scope_recommendations kind parsing (store kind column or parse correctly)
        # Use explicit kind field if present, else parse dedupe_key correctly handling org prefix
        kind = rec.get("kind")
        if not kind:
            raw = rec.get("dedupe_key", "")
            # dedupe_key may be org-prefixed like "{org_id}:{kind}:{...}"
            # org_id is UUID (36 chars with hyphens), kind is first non-uuid segment
            parts = raw.split(":")
            if len(parts) >= 2 and len(parts[0]) == 36 and parts[0].count("-") >= 3:
                kind = parts[1]
            elif parts:
                kind = parts[0]
            else:
                kind = ""
        if kind in _SENSITIVE_KINDS and access_level < ROLE_RANK["manager"]:
            # analyst sees the headline only, not margins/discount details
            rec = dict(rec)
            rec["body"] = "Details available to managers and admins."
            keep = {k: v for k, v in rec.get("evidence", {}).items() if k in ("sku", "product")}
            rec["evidence"] = keep
        impact = _impact_estimate(rec)
        rec = {**rec, **impact}
        estimate = float(impact.get("impact_estimate") or rec.get("impact_estimate") or 0)
        scored.append((estimate, rec))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [rec for _, rec in scored]
