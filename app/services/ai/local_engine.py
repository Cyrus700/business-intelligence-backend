"""Deterministic insight engine: answers BI questions from live warehouse data.

Used automatically when no LLM provider key is configured, or as a resilient
fallback when every provider fails. Produces the same professional, numbered
markdown style the LLM is prompted to use, so the UX is consistent.
"""

from datetime import date, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.ml import Anomaly, Forecast, MlModel
from app.services.ai.context import npr
from app.services.ai.intents import Intent
from app.services.analytics.queries import (
    Filters,
    expenses_by_category,
    inventory_levels,
    kpi_summary,
    monthly_pnl,
    sales_by_dimension,
)

WINDOW_DAYS = 30
MORE_LIMIT = 5
WINDOW_SHIFTS = 2  # how many empty windows to skip when data is stale


async def _kpi_map(db: AsyncSession, f: Filters) -> dict[str, dict[str, Any]]:
    return {c["metric"]: c for c in await kpi_summary(db, f)}


def _period_label(f: Filters) -> str:
    """Human label for the window actually being queried."""
    today = date.today()
    expected_start = today - timedelta(days=WINDOW_DAYS - 1)
    if f.date_from == expected_start and f.date_to == today:
        return f"last {WINDOW_DAYS} days"
    return f"{f.date_from} → {f.date_to}"


async def _resolve_window(db: AsyncSession, today: date | None = None) -> Filters:
    """Most recent {WINDOW_DAYS}-day window that actually contains data.

    Skips back up to WINDOW_SHIFTS empty windows so the assistant answers with
    the latest real numbers instead of claiming the business has no data.
    """
    today = today or date.today()
    for shift in range(WINDOW_SHIFTS + 1):
        span = shift * WINDOW_DAYS
        f = Filters(
            date_from=today - timedelta(days=WINDOW_DAYS - 1 + span),
            date_to=today - timedelta(days=span),
        )
        try:
            cards = await _kpi_map(db, f)
        except Exception:
            return f
        rev = cards.get("revenue", {}).get("value") or 0
        exp = cards.get("expense_total", {}).get("value") or 0
        if float(rev) > 0 or float(exp) > 0:
            return f
    return Filters(date_from=today - timedelta(days=WINDOW_DAYS - 1), date_to=today)


async def local_answer(db: AsyncSession, question: str, intent: Intent) -> str:
    f = await _resolve_window(db)

    try:
        handler = _HANDLERS.get(intent)
        if handler:
            return await handler(db, f, question)
    except Exception:
        # Fall through to generic answer rather than surfacing a 500 to the chat UI.
        pass

    return await _generic(db, f)


# ── handlers ────────────────────────────────────────────────────────────

async def _revenue(db: AsyncSession, f: Filters, q: str) -> str:
    cards = await _kpi_map(db, f)
    rev = cards.get("revenue", {})
    value, change = rev.get("value"), rev.get("change_pct")
    if value is None or value == 0:
        return _no_data("revenue")
    rows = await sales_by_dimension(db, f, "product")
    top = rows[0] if rows else None
    lines = [
        f"### Revenue — {_period_label(f)}",
        "",
        f"- **Total revenue:** {npr(value)}",
    ]
    if change is not None:
        lines.append(f"- **Change vs previous period:** {change:+.1f}%")
    if top:
        lines.append(
            f"- **Top seller:** {top['key']} at {npr(top['revenue'])} "
            f"({top['share_pct']}% of sales)"
        )
    lines += [
        "",
        "**Suggested action:**",
        "- Watch the **Revenue** KPI card and **Revenue vs Expenses** chart on your dashboard",
        f"- Check {top['key'] if top else 'top product'} performance in the "
        "**Sales Explorer** panel",
    ]
    return "\n".join(lines)


async def _expenses(db: AsyncSession, f: Filters, q: str) -> str:
    cards = await _kpi_map(db, f)
    exp = cards.get("expense_total", {})
    value, change = exp.get("value"), exp.get("change_pct")
    if value is None or value == 0:
        return _no_data("expenses")
    cats = await expenses_by_category(db, f)
    lines = [
        f"### Expenses — {_period_label(f)}",
        "",
        f"- **Total expenses:** {npr(value)}",
    ]
    if change is not None:
        lines.append(f"- **Change vs previous period:** {change:+.1f}%")
    if cats:
        lines.append("")
        lines.append("**Top cost centres:**")
        for c in cats[:MORE_LIMIT]:
            lines.append(f"- {c['key']}: {npr(c['revenue'])} ({c['share_pct']}% of total)")
    lines += [
        "",
        "**Suggested action:**",
        f"- Review the top cost centre ({cats[0]['key'] if cats else 'n/a'}) in the "
        "**Expense Breakdown** panel",
        "- Compare against revenue in the **P&L** view to check margin pressure",
    ]
    return "\n".join(lines)


async def _profit(db: AsyncSession, f: Filters, q: str) -> str:
    cards = await _kpi_map(db, f)
    rev = cards.get("revenue", {}).get("value")
    exp = cards.get("expense_total", {}).get("value")
    margin = cards.get("gross_margin", {}).get("value")
    if rev is None or exp is None or margin is None or (rev == 0 and exp == 0):
        return _no_data("profitability")
    net = float(margin) - float(exp)
    margin_pct = (float(margin) / rev * 100) if rev else 0
    health = (
        "healthy" if margin_pct >= 40 else "moderate" if margin_pct >= 25 else "under pressure"
    )
    lines = [
        f"### Profitability — {_period_label(f)}",
        "",
        f"- **Revenue:** {npr(rev)}",
        f"- **Gross margin:** {npr(margin)} ({margin_pct:.1f}% of revenue)",
        f"- **Expenses:** {npr(exp)}",
        f"- **Net profit:** {npr(net)}",
        "",
        f"Gross margin is {health} at {margin_pct:.1f}%.",
    ]
    return "\n".join(lines)


async def _forecast(db: AsyncSession, f: Filters, q: str) -> str:
    model = (
        await db.execute(
            select(MlModel)
            .where(MlModel.target == "revenue_daily", MlModel.is_active.is_(True))
            .order_by(MlModel.trained_at.desc())
        )
    ).scalar_one_or_none()
    if not model:
        return (
            "No active forecast model found. Train one from the **ML / Forecast** section, "
            "then ask me again — or check the **Revenue Forecast** panel."
        )
    rows = (
        (
            await db.execute(
                select(Forecast)
                .where(Forecast.model_id == model.id)
                .order_by(Forecast.forecast_date)
                .limit(WINDOW_DAYS)
            )
        )
        .scalars()
        .all()
    )
    if not rows:
        return (
            "The forecast model has no projections yet. "
            "Retrain it from the **Forecast** section."
        )
    total = sum(float(r.yhat) for r in rows)
    avg = total / len(rows)
    lo = sum(float(r.yhat_lower) if r.yhat_lower is not None else float(r.yhat) for r in rows)
    hi = sum(float(r.yhat_upper) if r.yhat_upper is not None else float(r.yhat) for r in rows)
    metrics = (model.metrics or {}) if model.metrics else {}
    acc = metrics.get("mape") or metrics.get("mae")
    acc_txt = (
        f" (historical MAPE/error ≈ {float(acc):.2f})" if acc is not None else ""
    )
    return "\n".join(
        [
            f"### Revenue Forecast — next {len(rows)} days",
            "",
            f"- **Projected total:** {npr(total)}",
            f"- **Daily average:** ~{npr(avg)}",
            f"- **Confidence band:** {npr(lo)} – {npr(hi)}",
            f"- **Model:** {model.model_type} v{model.version}{acc_txt}",
            "",
            "**Planning tip:** use the lower bound for conservative cash-flow planning and the "
            "upper bound for capacity decisions. Open the **Forecast** panel for the full curve.",
        ]
    )


async def _inventory(db: AsyncSession, f: Filters, q: str) -> str:
    low = await inventory_levels(db, below_reorder_only=True)
    if not low:
        return (
            "Good news — **no products are below their reorder level** right now. "
            "Inventory looks healthy."
        )
    urgent = [r for r in low if r["quantity_on_hand"] <= r["reorder_level"] * 0.5]
    lines = [
        f"### Inventory Alert — {len(low)} product(s) below reorder level",
        "",
        "| Product | On hand | Reorder | Status |",
        "|---|---|---|---|",
    ]
    shown = low[:10]
    for r in shown:
        status = "⚠️ Critical" if r in urgent else "Low"
        name = r["product"] or r["sku"]
        lines.append(
            f"| {name} | {r['quantity_on_hand']:,.0f} "
            f"| {r['reorder_level']:,.0f} | {status} |"
        )
    if len(low) > len(shown):
        lines.append(f"\n…and {len(low) - len(shown)} more.")
    lines += [
        "",
        "**Recommended action:** place reorder POs for critical items first — "
        "see the **Low Stock** panel on your dashboard.",
    ]
    return "\n".join(lines)


async def _anomalies(db: AsyncSession, f: Filters, q: str) -> str:
    rows = (
        (
            await db.execute(
                select(Anomaly)
                .where(Anomaly.status == "open")
                .order_by(Anomaly.detected_at.desc())
                .limit(20)
            )
        )
        .scalars()
        .all()
    )
    if not rows:
        return "No **open anomalies** detected in the current window — operations look stable. ✔"
    lines = [
        f"### Anomaly Alerts — {len(rows)} open",
        "",
        "| Metric | Observed | Expected | Deviation | Severity |",
        "|---|---|---|---|---|",
    ]
    sev_icons = {"high": "🔴 High", "medium": "🟠 Medium", "low": "🟡 Low"}
    for a in rows[:10]:
        expected = npr(a.expected_value) if a.expected_value is not None else "—"
        dev = (
            f"{(float(a.deviation_score) * 100):+.1f}%"
            if a.deviation_score is not None
            else "—"
        )
        sev = sev_icons.get(a.severity, a.severity)
        metric = a.metric.replace("_", " ")
        lines.append(
            f"| {metric} | {npr(a.observed_value)} | {expected} | {dev} | {sev} |"
        )
    lines += [
        "",
        "**Suggested action:** acknowledge high-severity alerts and review the "
        "**Anomaly Alerts** panel for context behind each spike.",
    ]
    return "\n".join(lines)


async def _products(db: AsyncSession, f: Filters, q: str) -> str:
    rows = await sales_by_dimension(db, f, "product")
    if not rows:
        return _no_data("product sales")
    lines = [
        f"### Top Products — {_period_label(f)}",
        "",
        "| Product | Revenue | Share | Orders |",
        "|---|---|---|---|",
    ]
    for p in rows[:8]:
        lines.append(f"| {p['key']} | {npr(p['revenue'])} | {p['share_pct']}% | {p['orders']} |")
    lines.append("")
    lines.append(
        f"**{rows[0]['key']}** is your best seller at {npr(rows[0]['revenue'])} — "
        f"consider promoting it further or protecting its supply."
    )
    return "\n".join(lines)


async def _dimension(
    db: AsyncSession, f: Filters, q: str, dim: str, title: str, tip: str
) -> str:
    rows = await sales_by_dimension(db, f, dim)
    if not rows:
        return _no_data(f"{title.lower()} sales")
    lines = [
        f"### {title} — {_period_label(f)}",
        "",
        "| Segment | Revenue | Share | Orders |",
        "|---|---|---|---|",
    ]
    for r in rows[:8]:
        lines.append(
            f"| {r['key']} | {npr(r['revenue'])} | {r['share_pct']}% | {r['orders']} |"
        )
    lines += ["", f"**{tip}**"]
    return "\n".join(lines)


async def _channels(db: AsyncSession, f: Filters, q: str) -> str:
    tip = (
        "Channel mix matters — focus spend on your highest-share channel "
        "while fixing underperformers."
    )
    return await _dimension(db, f, q, "channel", "Sales by Channel", tip)


async def _regions(db: AsyncSession, f: Filters, q: str) -> str:
    tip = (
        "Geographic gaps may signal demand, supply, or marketing issues "
        "worth investigating."
    )
    return await _dimension(db, f, q, "region", "Sales by Region", tip)


async def _compare(db: AsyncSession, f: Filters, q: str) -> str:
    pnl = await monthly_pnl(db, f)
    if not pnl:
        return _no_data("comparative monthly data")
    latest, prev = pnl[-1], pnl[-2] if len(pnl) > 1 else None
    lines = [
        f"### Latest Month vs Previous — {latest['month']:%b %Y}",
        "",
        f"- **Revenue:** {npr(latest['revenue'])}",
    ]
    if prev:
        rev_prev = float(prev["revenue"]) or 1.0
        exp_prev = float(prev["expenses"]) or 1.0
        d_rev = (float(latest["revenue"]) - float(prev["revenue"])) / rev_prev * 100
        d_exp = (float(latest["expenses"]) - float(prev["expenses"])) / exp_prev * 100
        lines.append(f"  - vs previous month: {d_rev:+.1f}%")
        lines.append(f"- **Expenses:** {npr(latest['expenses'])} ({d_exp:+.1f}% vs previous)")
    lines.append(f"- **Net profit:** {npr(latest['net'])}")
    return "\n".join(lines)


# ── social / help intents ───────────────────────────────────────────────

async def _greeting(db: AsyncSession, f: Filters, q: str) -> str:
    if any(
        k in q.lower()
        for k in ("how are you", "how r u", "how are u", "hru", "how's it going", "how is it going")
    ):
        return (
            "I'm doing great, thank you for asking! 🙌 How can I help you with your "
            "business data today?\n\n"
            "Ask me about **revenue**, **expenses**, **forecasts**, **inventory**, "
            "**anomalies**, or **top products** — I'll pull the real numbers."
        )
    return (
        "Hi! 👋 I'm Insightful AI, your business intelligence co-pilot.\n\n"
        "Ask me anything about your **live data** — revenue, expenses, forecasts, "
        "inventory, anomalies, or product performance — and I'll answer with real "
        "numbers, never guesses.\n\n"
        "Try: *\"What's our revenue trend this month?\"*"
    )


async def _help(db: AsyncSession, f: Filters, q: str) -> str:
    return (
        "Here's everything I can do for you:\n\n"
        "**📊 Revenue & sales** — trends, totals, comparisons vs last period\n"
        "**💸 Expenses** — totals and top cost centres\n"
        "**📈 Forecasts** — 30-day revenue projections with confidence bands\n"
        "**📦 Inventory** — products below reorder level, restock priorities\n"
        "**🚨 Anomalies** — open alerts with severity and deviation\n"
        "**🏆 Products / Channels / Regions** — top performers and share\n"
        "**💰 Profitability** — margin and net profit\n\n"
        "I always answer from your **live dashboard data**, never from guesses."
    )


async def _thanks(db: AsyncSession, f: Filters, q: str) -> str:
    return (
        "You're welcome! 🙌 If you need numbers on revenue, expenses, inventory, "
        "forecasts, or anomalies, I'm here. Anything else you'd like to check?"
    )


async def _generic(db: AsyncSession, f: Filters) -> str:
    cards = await _kpi_map(db, f)
    rev = cards.get("revenue", {}).get("value")
    exp = cards.get("expense_total", {}).get("value")
    if rev is None and exp is None:
        return (
            "I couldn't match that question to a specific metric — but here's a live overview:\n\n"
            "The **Revenue** and **Expenses** KPIs at the top of your dashboard, the "
            "**Forecast** panel, **Low Stock** and **Anomaly Alerts** panels are all "
            "interactive. Ask me about any of them and I'll pull the actual numbers."
        )

    lines = [
        "Here's a quick snapshot of your business:",
        "",
        f"- **Revenue (30d):** {npr(rev) if rev is not None else '—'}",
        f"- **Expenses (30d):** {npr(exp) if exp is not None else '—'}",
    ]
    try:
        low = await inventory_levels(db, below_reorder_only=True)
        if low:
            lines.append(f"- **Stock alerts:** {len(low)} product(s) below reorder level")
    except Exception:
        pass
    try:
        open_anoms = (
            (
                await db.execute(
                    select(Anomaly)
                    .where(Anomaly.status == "open")
                    .order_by(Anomaly.detected_at.desc())
                    .limit(20)
                )
            )
            .scalars()
            .all()
        )
        if open_anoms:
            lines.append(f"- **Anomalies:** {len(open_anoms)} open alert(s) need review")
    except Exception:
        pass
    lines += [
        "",
        "To go deeper, ask about **forecasts**, **inventory**, **anomalies**, "
        "**top products**, or **expense categories**.",
    ]
    return "\n".join(lines)


def _no_data(what: str) -> str:
    return (
        f"No {what} data is available yet. Connect a data source or check the "
        "**Data Sources / ETL** section so I can answer with real numbers."
    )


_HANDLERS: dict[Intent, Any] = {
    Intent.REVENUE: _revenue,
    Intent.EXPENSES: _expenses,
    Intent.PROFIT: _profit,
    Intent.FORECAST: _forecast,
    Intent.INVENTORY: _inventory,
    Intent.ANOMALIES: _anomalies,
    Intent.PRODUCTS: _products,
    Intent.CHANNELS: _channels,
    Intent.REGIONS: _regions,
    Intent.COMPARE: _compare,
    Intent.GREETING: _greeting,
    Intent.HELP: _help,
    Intent.THANKS: _thanks,
    Intent.CAPABILITIES: _help,
}
