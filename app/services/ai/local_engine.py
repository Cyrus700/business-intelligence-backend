"""Deterministic insight engine: answers BI questions from live warehouse data.

Used automatically when no LLM provider key is configured, or as a resilient
fallback when every provider fails. Produces the same professional, numbered
markdown style the LLM is prompted to use, so the UX is consistent.
"""

import logging
import re
from datetime import date, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.clock import business_today
from app.models.ml import Anomaly, Forecast, MlModel
from app.services.ai.context import npr
from app.services.ai.dates import ParsedPeriod, parse_period
from app.services.ai.intents import Intent
from app.services.analytics.queries import (
    Filters,
    data_coverage,
    expenses_by_category,
    inventory_levels,
    kpi_summary,
    monthly_pnl,
    sales_by_dimension,
)

logger = logging.getLogger(__name__)

WINDOW_DAYS = 30
MORE_LIMIT = 5
WINDOW_SHIFTS = 2  # how many empty windows to skip when data is stale


async def _kpi_map(db: AsyncSession, f: Filters) -> dict[str, dict[str, Any]]:
    return {c["metric"]: c for c in await kpi_summary(db, f)}


def _org_for_user(user) -> Any | None:
    if user is None:
        return None
    if getattr(user, "is_super_admin", False):
        return None
    return getattr(user, "org_id", None)


def _period_label(f: Filters) -> str:
    """Human label for the window actually being queried."""
    label = _LABELS.get((f.date_from, f.date_to))
    if label:
        return label
    today = business_today()
    expected_start = today - timedelta(days=WINDOW_DAYS - 1)
    if f.date_from == expected_start and f.date_to == today:
        return f"last {WINDOW_DAYS} days"
    if f.date_from == f.date_to:
        return f.date_from.strftime("%-d %b %Y")
    return f"{f.date_from} → {f.date_to}"


# Labels for windows the question named explicitly ("yesterday", "June 2026"),
# so the reply quotes the user's own framing back instead of raw ISO bounds.
_LABELS: dict[tuple[date, date], str] = {}


def _remember_label(f: Filters, label: str) -> None:
    _LABELS[(f.date_from, f.date_to)] = label
    if len(_LABELS) > 256:  # bounded: this is a formatting cache, not state
        _LABELS.clear()


async def _resolve_window(db: AsyncSession, today: date | None = None, org_id=None) -> Filters:
    """Most recent {WINDOW_DAYS}-day window that actually contains data.

    Skips back up to WINDOW_SHIFTS empty windows so the assistant answers with
    the latest real numbers instead of claiming the business has no data.
    """
    today = today or business_today()
    for shift in range(WINDOW_SHIFTS + 1):
        span = shift * WINDOW_DAYS
        f = Filters(
            date_from=today - timedelta(days=WINDOW_DAYS - 1 + span),
            date_to=today - timedelta(days=span),
            org_id=org_id,
        )
        try:
            cards = await _kpi_map(db, f)
        except Exception:
            return f
        rev = cards.get("revenue", {}).get("value") or 0
        exp = cards.get("expense_total", {}).get("value") or 0
        if float(rev) > 0 or float(exp) > 0:
            return f
    return Filters(date_from=today - timedelta(days=WINDOW_DAYS - 1), date_to=today, org_id=org_id)


async def local_answer(db: AsyncSession, question: str, intent: Intent, org_id=None, user=None) -> str:
    """Answer from live warehouse data without an LLM.

    A period named in the question wins over the rolling default, and is never
    silently widened — otherwise "revenue on 10 June" and "revenue on 1 Jan
    2019" both collapse onto the same 30-day window and return byte-identical
    replies.
    """
    # Resolve org_id from user if not explicitly given
    if org_id is None and user is not None:
        org_id = _org_for_user(user)
    asked = parse_period(question)
    if asked is not None:
        f = Filters(date_from=asked.start, date_to=asked.end, org_id=org_id)
        _remember_label(f, asked.label)
        outside = await _outside_coverage(db, asked, org_id=org_id)
        if outside:
            return outside
    else:
        f = await _resolve_window(db, org_id=org_id)

    try:
        handler = _HANDLERS.get(intent)
        if handler:
            return await handler(db, f, question)
    except Exception:
        # Fall through to the generic answer rather than surfacing a 500 to the
        # chat UI. The rollback matters: a failed statement aborts the whole
        # Postgres transaction, so _generic's own queries would fail too.
        logger.warning("local handler for %s failed", intent, exc_info=True)
        await _rollback(db)

    return await _generic(db, f)


async def _rollback(db: AsyncSession) -> None:
    try:
        await db.rollback()
    except Exception:  # pragma: no cover - session already unusable
        logger.debug("rollback failed", exc_info=True)


async def _outside_coverage(db: AsyncSession, asked: ParsedPeriod, org_id=None) -> str | None:
    """Explain a window the warehouse simply has no data for.

    Reporting रू 0 for a date that was never loaded is the single most
    misleading thing this assistant can do, so it is called out explicitly.
    """
    try:
        coverage = await data_coverage(db, org_id=org_id)
    except Exception:
        # e.g. the ingested_at column is missing because migrations are behind.
        logger.warning("data_coverage unavailable", exc_info=True)
        await _rollback(db)
        return None
    first, last = coverage.get("first_date"), coverage.get("last_date")
    if first is None or last is None:
        return (
            "There's no data in the warehouse yet, so I can't report on "
            f"**{asked.label}**. Upload a file or connect a data source first."
        )
    if asked.end < first or asked.start > last:
        return (
            f"I have no data for **{asked.label}**. The warehouse currently covers "
            f"**{first:%-d %b %Y} → {last:%-d %b %Y}**, so this is missing data rather "
            "than zero sales.\n\n"
            "**Suggested action:** upload the file covering that period from the "
            "**Data** page, then ask me again."
        )
    return None


def _revenue_action(value: float, change: float | None, top: dict | None) -> str:
    """A next step that changes with the numbers, not a fixed sign-off."""
    name = top["key"] if top else None
    share = float(top["share_pct"]) if top else 0.0
    if change is not None and change <= -15:
        return (
            f"Revenue fell {abs(change):.1f}%. Open **Analytics → by channel** for this "
            f"period to find where the drop came from"
            + (f", starting with {name}, your largest line." if name else ".")
        )
    if share >= 50 and name:
        return (
            f"{name} alone is {share:.1f}% of revenue — that is heavy concentration. "
            "Check its stock cover in **Inventory** before it becomes a single point of failure."
        )
    if change is not None and change >= 15 and name:
        return (
            f"Growth of {change:+.1f}% is led by {name}. Confirm you have inventory "
            "to sustain it in the **Low Stock** panel."
        )
    if name:
        return f"Review {name} ({share:.1f}% of sales) in **Analytics → by product**."
    return "Upload more sales history to make this period comparable."


def _expense_action(value: float, change: float | None, cats: list[dict]) -> str:
    if not cats:
        return "No expense categories are recorded for this period — check the finance upload."
    top = cats[0]
    share = float(top["share_pct"])
    if change is not None and change >= 15:
        return (
            f"Spend rose {change:+.1f}%, and {top['key']} is {share:.1f}% of it. "
            "Compare against revenue in the **P&L** view before it eats the margin."
        )
    if share >= 50:
        return (
            f"{top['key']} is {share:.1f}% of all spend — a single renegotiation there "
            "moves the bottom line more than trimming everything else."
        )
    return f"{top['key']} leads spend at {npr(top['revenue'])}; review it in **P&L**."


# ── handlers ────────────────────────────────────────────────────────────


async def _revenue(db: AsyncSession, f: Filters, q: str) -> str:
    cards = await _kpi_map(db, f)
    rev = cards.get("revenue", {})
    value, change = rev.get("value"), rev.get("change_pct")
    if value is None or value == 0:
        return _no_data("revenue", _period_label(f))
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
        lines.append(f"- **Top seller:** {top['key']} at {npr(top['revenue'])} ({top['share_pct']}% of sales)")
    lines += ["", f"**Suggested action:** {_revenue_action(value, change, top)}"]
    return "\n".join(lines)


async def _expenses(db: AsyncSession, f: Filters, q: str) -> str:
    cards = await _kpi_map(db, f)
    exp = cards.get("expense_total", {})
    value, change = exp.get("value"), exp.get("change_pct")
    if value is None or value == 0:
        return _no_data("expenses", _period_label(f))
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
    lines += ["", f"**Suggested action:** {_expense_action(value, change, cats)}"]
    return "\n".join(lines)


async def _profit(db: AsyncSession, f: Filters, q: str) -> str:
    cards = await _kpi_map(db, f)
    rev = cards.get("revenue", {}).get("value")
    exp = cards.get("expense_total", {}).get("value")
    margin = cards.get("gross_margin", {}).get("value")
    if rev is None or exp is None or margin is None or (rev == 0 and exp == 0):
        return _no_data("profitability", _period_label(f))
    net = float(margin) - float(exp)
    margin_pct = (float(margin) / rev * 100) if rev else 0
    health = "healthy" if margin_pct >= 40 else "moderate" if margin_pct >= 25 else "under pressure"
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
    mq = select(MlModel).where(MlModel.target == "revenue_daily", MlModel.is_active.is_(True))
    if f.org_id is not None:
        mq = mq.where(MlModel.org_id == f.org_id)
    model = (await db.execute(mq.order_by(MlModel.trained_at.desc()))).scalar_one_or_none()
    if not model:
        return (
            "No active forecast model found. Train one from the **ML / Forecast** section, "
            "then ask me again — or check the **Revenue Forecast** panel."
        )
    fq = select(Forecast).where(Forecast.model_id == model.id).order_by(Forecast.forecast_date).limit(WINDOW_DAYS)
    if f.org_id is not None:
        fq = fq.where(Forecast.org_id == f.org_id)
    rows = (await db.execute(fq)).scalars().all()
    if not rows:
        return "The forecast model has no projections yet. Retrain it from the **Forecast** section."
    total = sum(float(r.yhat) for r in rows)
    avg = total / len(rows)
    lo = sum(float(r.yhat_lower) if r.yhat_lower is not None else float(r.yhat) for r in rows)
    hi = sum(float(r.yhat_upper) if r.yhat_upper is not None else float(r.yhat) for r in rows)
    metrics = (model.metrics or {}) if model.metrics else {}
    acc = metrics.get("mape") or metrics.get("mae")
    acc_txt = f" (historical MAPE/error ≈ {float(acc):.2f})" if acc is not None else ""
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
    low = await inventory_levels(db, below_reorder_only=True, org_id=f.org_id)
    if not low:
        return "Good news — **no products are below their reorder level** right now. Inventory looks healthy."
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
        lines.append(f"| {name} | {r['quantity_on_hand']:,.0f} | {r['reorder_level']:,.0f} | {status} |")
    if len(low) > len(shown):
        lines.append(f"\n…and {len(low) - len(shown)} more.")
    lines += [
        "",
        "**Recommended action:** place reorder POs for critical items first — "
        "see the **Low Stock** panel on your dashboard.",
    ]
    return "\n".join(lines)


async def _anomalies(db: AsyncSession, f: Filters, q: str) -> str:
    aq = select(Anomaly).where(Anomaly.status == "open").order_by(Anomaly.detected_at.desc()).limit(20)
    if f.org_id is not None:
        aq = aq.where(Anomaly.org_id == f.org_id)
    rows = (await db.execute(aq)).scalars().all()
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
        dev = f"{(float(a.deviation_score) * 100):+.1f}%" if a.deviation_score is not None else "—"
        sev = sev_icons.get(a.severity, a.severity)
        metric = a.metric.replace("_", " ")
        lines.append(f"| {metric} | {npr(a.observed_value)} | {expected} | {dev} | {sev} |")
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


async def _dimension(db: AsyncSession, f: Filters, q: str, dim: str, title: str, tip: str) -> str:
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
        lines.append(f"| {r['key']} | {npr(r['revenue'])} | {r['share_pct']}% | {r['orders']} |")
    lines += ["", f"**{tip}**"]
    return "\n".join(lines)


async def _channels(db: AsyncSession, f: Filters, q: str) -> str:
    tip = "Channel mix matters — focus spend on your highest-share channel while fixing underperformers."
    return await _dimension(db, f, q, "channel", "Sales by Channel", tip)


async def _regions(db: AsyncSession, f: Filters, q: str) -> str:
    tip = "Geographic gaps may signal demand, supply, or marketing issues worth investigating."
    return await _dimension(db, f, q, "region", "Sales by Region", tip)


async def _compare(db: AsyncSession, f: Filters, q: str) -> str:
    # "compare 10 June and 12 June" names two specific periods — answer about
    # those, not about whatever two months happen to be latest.
    named = _named_periods(q)
    if named:
        return await _compare_periods(db, named, org_id=f.org_id)

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


def _named_periods(question: str) -> list[ParsedPeriod] | None:
    """Two periods explicitly named in a comparison question, in order.

    Splits on the comparison connective first so each side is parsed on its own
    — parse_period() over the whole string would collapse "10 June vs 12 June"
    into a single 10→12 span and lose the comparison.
    """
    parts = re.split(r"\b(?:vs\.?|versus|against|compared to|and)\b", question, maxsplit=1)
    if len(parts) != 2:
        return None
    left, right = (parse_period(part) for part in parts)
    if left is None or right is None or (left.start, left.end) == (right.start, right.end):
        return None
    return [left, right]


async def _compare_periods(db: AsyncSession, periods: list[ParsedPeriod], org_id=None) -> str:
    rows = []
    for period in periods:
        f = Filters(date_from=period.start, date_to=period.end, org_id=org_id)
        cards = await _kpi_map(db, f)
        rows.append(
            {
                "label": period.label,
                "revenue": float(cards.get("revenue", {}).get("value") or 0),
                "orders": float(cards.get("orders", {}).get("value") or 0),
            }
        )

    first, second = rows[0], rows[1]
    base = first["revenue"] or None
    delta = ((second["revenue"] - first["revenue"]) / base * 100) if base else None
    lines = [
        f"### {first['label']} vs {second['label']}",
        "",
        "| Period | Revenue | Orders |",
        "|---|---|---|",
        f"| {first['label']} | {npr(first['revenue'])} | {first['orders']:,.0f} |",
        f"| {second['label']} | {npr(second['revenue'])} | {second['orders']:,.0f} |",
        "",
    ]
    if delta is None:
        lines.append(f"{first['label']} had no revenue, so there is no percentage to compare against.")
    else:
        direction = "up" if delta >= 0 else "down"
        lines.append(
            f"Revenue was {direction} **{abs(delta):.1f}%** in {second['label']} "
            f"({npr(second['revenue'] - first['revenue'])} difference)."
        )
        better = second if second["revenue"] >= first["revenue"] else first
        lines += [
            "",
            f"**Suggested action:** {better['label']} was the stronger period — "
            "open **Analytics → by channel** for it to see which channel drove the gap.",
        ]
    return "\n".join(lines)


# ── social / help intents ───────────────────────────────────────────────


async def _greeting(db: AsyncSession, f: Filters, q: str) -> str:
    if any(k in q.lower() for k in ("how are you", "how r u", "how are u", "hru", "how's it going", "how is it going")):
        return (
            "I'm doing great, thank you for asking! 🙌 How can I help you with your "
            "business data today?\n\n"
            "Ask me about **revenue**, **expenses**, **forecasts**, **inventory**, "
            "**anomalies**, or **top products** — I'll pull the real numbers."
        )
    return (
        "Hi! 👋 I'm InsightFlow AI, your business intelligence co-pilot.\n\n"
        "Ask me anything about your **live data** — revenue, expenses, forecasts, "
        "inventory, anomalies, or product performance — and I'll answer with real "
        "numbers, never guesses.\n\n"
        'Try: *"What\'s our revenue trend this month?"*'
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


def _extract_platform_status(q: str) -> str | None:
    """Extract status filter from free-text, typo-tolerant.

    Returns one of: approved | pending | rejected | legacy | personal | None
    Handles misspellings: aprrved, rejeect, rejefct, pendng etc via substring roots.
    """
    ql = q.lower()
    # approved variants: appr, aprr, aprv, approov — catch all appr/aprr/aprv roots
    if any(root in ql for root in ("appr", "aprv", "aprr", "approv")):
        return "approved"
    if "rej" in ql:  # covers rejected, rejeect, rejefct, rejeefect, rejact etc
        return "rejected"
    if "pend" in ql:  # pending, pendng, pendding
        return "pending"
    if "legacy" in ql:
        return "legacy"
    if "personal" in ql:
        return "personal"
    return None


async def _platform(db: AsyncSession, f: Filters, q: str) -> str:
    """Live platform stats — precise answer, no KPI dump.

    If question contains a status (approved/pending/rejected with typos), returns
    a single-number precise answer for that status. Otherwise returns total
    breakdown. Always concise.
    """
    try:
        from sqlalchemy import func as _func
        from sqlalchemy import select as _select

        from app.models.identity import Organization, Profile

        is_super = f.org_id is None
        ql = q.lower()
        wants_list = any(w in ql for w in ("list", "show", "detail", "names", "which"))
        status_filter = _extract_platform_status(q)
        # Status-specific precise path — super-admin only; isolated users get isolation note with their own status
        if status_filter and is_super:
            # precise single-status count, typo-tolerant
            if status_filter in ("approved", "pending", "rejected"):
                cnt = (
                    await db.execute(
                        _select(_func.count()).select_from(Organization).where(Organization.status == status_filter)
                    )
                ).scalar() or 0
                total = (await db.execute(_select(_func.count()).select_from(Organization))).scalar() or 0
                lines = [
                    f"**{cnt}** businesses are **{status_filter}** (of **{total}** total, live).",
                ]
                if wants_list:
                    rows = (
                        await db.execute(
                            _select(Organization)
                            .where(Organization.status == status_filter)
                            .order_by(Organization.created_at.desc())
                            .limit(15)
                        )
                    ).scalars().all()
                    if rows:
                        lines.append("")
                        lines.append(f"| Business | Status | Created |")
                        lines.append(f"|---|---|---|")
                        for o in rows:
                            created = o.created_at.date().isoformat() if getattr(o, "created_at", None) else "—"
                            lines.append(f"| {o.name} | {o.status} | {created} |")
                    else:
                        lines.append(f"No businesses with status `{status_filter}` right now.")
                else:
                    lines.append(f"Ask `list {status_filter} businesses` to see names.")
                return "\n".join(lines)
            if status_filter == "legacy":
                cnt = (
                    await db.execute(_select(_func.count()).select_from(Organization).where(Organization.is_legacy.is_(True)))
                ).scalar() or 0
                return f"**{cnt}** businesses are **legacy** workspaces (live)."
            if status_filter == "personal":
                cnt = (
                    await db.execute(_select(_func.count()).select_from(Organization).where(Organization.is_personal.is_(True)))
                ).scalar() or 0
                return f"**{cnt}** businesses are **personal** workspaces (live)."

        # General total breakdown — concise, no revenue/top-products
        if is_super:
            org_total = (await db.execute(_select(_func.count()).select_from(Organization))).scalar() or 0
            approved = (
                await db.execute(_select(_func.count()).select_from(Organization).where(Organization.status == "approved"))
            ).scalar() or 0
            pending = (
                await db.execute(_select(_func.count()).select_from(Organization).where(Organization.status == "pending"))
            ).scalar() or 0
            rejected = (
                await db.execute(_select(_func.count()).select_from(Organization).where(Organization.status == "rejected"))
            ).scalar() or 0
            # Validation: total should equal sum; if gap, explain
            calc = approved + pending + rejected
            lines = [
                f"There are **{org_total}** businesses registered (live).",
                f"- Approved: **{approved}** · Pending: **{pending}** · Rejected: **{rejected}**",
            ]
            if calc != org_total:
                lines.append(f"  — note: {org_total - calc} in other/unknown status")
            if wants_list:
                rows = (
                    await db.execute(_select(Organization).order_by(Organization.created_at.desc()).limit(15))
                ).scalars().all()
                if rows:
                    lines.append("")
                    lines.append("| Business | Status | Created |")
                    lines.append("|---|---|---|")
                    for o in rows:
                        created = o.created_at.date().isoformat() if getattr(o, "created_at", None) else "—"
                        lines.append(f"| {o.name} | {o.status} | {created} |")
            else:
                lines.append("Ask `list businesses` to see names.")
            return "\n".join(lines)
        # Isolated user — they asked "how many businesses are there?" The
        # truthful answer is one (their own), plus isolation explanation.
        ql = q.lower()
        wants_users = any(k in ql for k in ("users", "members"))
        if wants_users:
            from sqlalchemy import func as _func2
            from sqlalchemy import select as _select2

            from app.models.identity import Profile as _P

            org_users = (
                await db.execute(_select2(_func2.count()).select_from(_P).where(_P.org_id == f.org_id))
            ).scalar() or 0
            org = await db.get(Organization, f.org_id) if f.org_id else None
            name = org.name if org else "your workspace"
            return (
                f"There are **{org_users}** user(s) in **{name}** (your isolated workspace).\n\n"
                f"- **Your scope:** isolated — you only see this business's data.\n"
                f"- **Platform total:** not visible from a business account. A Platform Super-Admin can see all {org_users and '' or ''}workspaces.\n\n"
                "**Suggested action:** open **Users** to see everyone in your workspace."
            )
        # Business count for isolated user
        org = await db.get(Organization, f.org_id) if f.org_id else None
        if org:
            return (
                f"You're in **{org.name}** — your isolated workspace.\n\n"
                f"- **Workspaces you can see:** 1 (this one — data is strictly per-business).\n"
                f"- **Org ID:** `{org.id}` · status: {getattr(org, 'status', 'approved')} · "
                f"{'personal workspace' if getattr(org, 'is_personal', False) else 'business workspace'}\n"
                f"- **Platform total:** as a business user you only see your own business. "
                f"A Super-Admin sees the full directory via **Admin Center**.\n\n"
                f"**Suggested action:** ask about **{org.name}**'s revenue, expenses, forecasts, or inventory — I'll pull live numbers for this workspace."
            )
        return (
            "I couldn't find your business name — your account has no workspace assigned. Ask your admin for an invite or register a business.\n\n"
            "For privacy, business counts are only visible to Platform Super-Admins."
        )
    except Exception:
        logger.warning("_platform handler failed", exc_info=True)
        await _rollback(db)
        return "I couldn't count businesses just now — please try again. If this persists, check **Admin → Businesses** for the live list."


async def _users_handler(db: AsyncSession, f: Filters, q: str) -> str:
    # Delegate to platform handler — same live counting logic
    return await _platform(db, f, q)


async def _catalog_handler(db: AsyncSession, f: Filters, q: str) -> str:
    """Live data dictionary — what tables/datasets are available right now."""
    try:
        from sqlalchemy import text as _text

        # Dynamic discovery: ask postgres what tables actually exist right now
        tables = (
            await db.execute(
                _text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema='public' AND table_type='BASE TABLE' "
                    "AND table_name NOT LIKE 'alembic%' ORDER BY table_name"
                )
            )
        ).scalars().all()

        # Friendly descriptions for the warehouse; unknown tables get a generic line
        descriptions = {
            "organizations": "Businesses / workspaces (tenants)",
            "profiles": "Users / members (with org_id isolation)",
            "sales_transactions": "Sales fact — one row per line-item sale",
            "expenses": "Expenses fact",
            "inventory_levels": "Inventory snapshots",
            "products": "Product dimension (per-org SKU)",
            "customers": "Customer dimension",
            "kpi_snapshots": "Pre-aggregated KPIs (per business-day)",
            "kpi_definitions": "KPI metadata / targets",
            "ml_models": "Forecast models (per org, per target)",
            "forecasts": "Forecast projections",
            "anomalies": "Anomaly alerts",
            "insights": "Generated insights / recommendations",
            "alert_rules": "Alert rules",
            "notifications": "Notifications",
            "reports": "Generated reports",
            "data_sources": "Data sources / ETL origins",
            "raw_uploads": "Raw upload history",
            "etl_jobs": "ETL job runs",
            "roles": "Roles catalog",
            "permissions": "Permissions catalog",
            "role_permissions": "Role ↔ permission matrix",
            "audit_logs": "Audit trail",
            "ai_conversations": "AI chat conversations (per org)",
            "ai_messages": "AI chat messages (per org)",
        }

        lines = [
            "### Data Catalog — live, right now",
            "",
            "Everything below is a **real table in your database** at this moment. "
            "New tables appear here automatically — no code change needed.",
            "",
            f"- **Total tables:** {len(tables)} in schema `public`",
            "",
            "| Table | What it holds | Rows you can see* |",
            "|---|---|---|",
        ]
        for t in tables:
            # Row count scoped to caller — realtime, no cache
            count = None
            try:
                has_org = (
                    await db.execute(
                        _text(
                            "SELECT 1 FROM information_schema.columns "
                            "WHERE table_schema='public' AND table_name=:t AND column_name='org_id' LIMIT 1"
                        ),
                        {"t": t},
                    )
                ).scalar()
                if has_org and f.org_id is not None:
                    count = (
                        await db.execute(_text(f'SELECT COUNT(*) FROM "{t}" WHERE org_id = :oid'), {"oid": str(f.org_id)})
                    ).scalar()
                else:
                    # super-admin sees full platform count; isolated user sees their slice if column exists else global count
                    if has_org and f.org_id is None:
                        count = (await db.execute(_text(f'SELECT COUNT(*) FROM "{t}"'))).scalar()
                    elif not has_org:
                        count = (await db.execute(_text(f'SELECT COUNT(*) FROM "{t}"'))).scalar()
                    else:
                        count = 0
            except Exception:
                count = "—"
            desc = descriptions.get(t, "—")
            lines.append(f"| `{t}` | {desc} | {count} |")
        lines += [
            "",
            "* Row counts are **live** and scoped to you: a business user sees only their org's rows where `org_id` exists; a Super-Admin sees the platform total.",
            "",
            "**Suggested action:** ask me directly about any table — e.g. *“how many rows in sales_transactions for last 30 days?”*, *“sample 5 products”*, or *“how many businesses are registered?”* — I'll query it live.",
        ]
        return "\n".join(lines)
    except Exception:
        logger.warning("_catalog handler failed", exc_info=True)
        await _rollback(db)
        return "I couldn't read the data catalog just now — please try again. The **Data** page lists your connected sources."


async def _business(db: AsyncSession, f: Filters, q: str) -> str:
    # If the question is really a count/listing, delegate to the platform handler
    ql = q.lower()
    if any(k in ql for k in ("how many", "count", "total", "number of", "registered", "list", "directory", "all business", "all organization", "are there")):
        return await _platform(db, f, q)
    # Otherwise accurate single-workspace identity — resolve org name from DB, never guess
    org_id = f.org_id
    try:
        if org_id is not None:
            from app.models.identity import Organization

            org = await db.get(Organization, org_id)
            if org:
                return (
                    f"Your business / workspace is **{org.name}**.\n\n"
                    f"- **Workspace:** {org.name}\n"
                    f"- **Org ID:** `{org.id}`\n"
                    f"- **Your scope:** isolated — you only see this business's data.\n\n"
                    "Ask me about its **revenue, expenses, forecasts, inventory, anomalies** — I'll pull live numbers for this workspace."
                )
        # Fallback: super-admin or no org

        from app.models.identity import Organization

        # If super-admin with no org, list accessible orgs hint
        if org_id is None:
            return (
                "You're signed in as **Platform Super-Admin** (no single business). "
                "You can see all workspaces via the org switcher.\n\n"
                "If you expected a business name, sign in with a Business Admin / Manager / Analyst account for that workspace."
            )
        return "I couldn't find your business name — your account has no workspace assigned. Ask your admin for an invite or register a business."
    except Exception:
        logger.warning("_business handler failed", exc_info=True)
        await _rollback(db)
        return "Your business name is stored in your workspace settings. I couldn't read it just now — try again."


async def _update_handler(db: AsyncSession, f: Filters, q: str) -> str:
    """Live 'what's the update' — comprehensive snapshot for ANY vague/status question.

    This is the universal fallback: it answers 'whats the update', 'summary',
    'overview', and any UNKNOWN phrasing with a precise, live, scannable digest
    across ALL domains so no question ever gets a generic apology.
    """
    try:
        from sqlalchemy import func as _func
        from sqlalchemy import select as _sel
        from sqlalchemy import text as _text

        from app.models.identity import Organization

        # Use the window already resolved (last 30 days with data) for KPIs
        cards = await _kpi_map(db, f)
        period_label = _period_label(f)
        coverage = None
        try:
            coverage = await data_coverage(db, org_id=f.org_id)
        except Exception:
            pass

        # Platform counts (live, scoped)
        platform_line = ""
        try:
            if f.org_id is None:
                total_orgs = (await db.execute(_sel(_func.count()).select_from(Organization))).scalar() or 0
                platform_line = f"- **Businesses registered:** {total_orgs} (live platform total)"
            else:
                org = await db.get(Organization, f.org_id) if f.org_id else None
                if org:
                    platform_line = f"- **Workspace:** {org.name} — isolated, 1 business you can see"
        except Exception:
            pass

        lines = [
            f"### Live Update — {period_label}",
            "",
            f"Here's the precise picture **right now** ({business_today().isoformat()}), all numbers live from your warehouse:",
            "",
        ]
        if platform_line:
            lines.append(platform_line)
        if coverage and coverage.get("first_date"):
            lines.append(f"- **Data freshness:** {coverage['first_date']} → {coverage['last_date']} ({coverage['days_behind']} day(s) behind today), last upload {coverage['last_ingested_at']:%Y-%m-%d %H:%M} " if coverage.get("last_ingested_at") else f"- **Warehouse:** {coverage['first_date']} → {coverage['last_date']}")
        # KPIs with change
        for metric in ("revenue", "orders", "avg_order_value", "gross_margin", "expense_total"):
            c = cards.get(metric)
            if not c or c.get("value") is None:
                continue
            change = c.get("change_pct")
            trend = f" ({change:+.1f}% vs previous period)" if change is not None else ""
            label = metric.replace("_", " ").title()
            lines.append(f"- **{label}:** {npr(c['value'])}{trend}")

        # Top products (live)
        try:
            prods = await sales_by_dimension(db, f, "product")
            if prods:
                lines.append("")
                lines.append(f"**Top products ({period_label}):**")
                for p in prods[:3]:
                    lines.append(f"- {p['key']}: {npr(p['revenue'])} ({p['share_pct']}% share, {p['orders']} orders)")
        except Exception:
            pass

        # Expense categories
        try:
            exps = await expenses_by_category(db, f)
            if exps:
                lines.append("")
                lines.append("**Spend by category:**")
                for e in exps[:3]:
                    lines.append(f"- {e['key']}: {npr(e['revenue'])} ({e['share_pct']}% of expenses)")
        except Exception:
            pass

        # Inventory
        try:
            low = await inventory_levels(db, below_reorder_only=True, org_id=f.org_id)
            if low:
                lines.append(f"\n- **Inventory:** {len(low)} product(s) below reorder — e.g. {', '.join((r['product'] or r['sku']) for r in low[:3])}")
            else:
                lines.append("\n- **Inventory:** all SKUs above reorder level — healthy")
        except Exception:
            pass

        # Anomalies
        try:
            aq = _sel(Anomaly).where(Anomaly.status == "open").order_by(Anomaly.detected_at.desc()).limit(5)
            if f.org_id is not None:
                aq = aq.where(Anomaly.org_id == f.org_id)
            anoms = (await db.execute(aq)).scalars().all()
            if anoms:
                latest = ", ".join(f"{a.metric.replace('_',' ')} {npr(a.observed_value)}" for a in anoms[:2])
                lines.append(f"- **Anomalies:** {len(anoms)} open — latest: {latest}")
            else:
                lines.append("- **Anomalies:** none open — stable")
        except Exception:
            pass

        # Forecast snapshot
        try:
            mq = _sel(MlModel).where(MlModel.target == "revenue_daily", MlModel.is_active.is_(True))
            if f.org_id is not None:
                mq = mq.where(MlModel.org_id == f.org_id)
            model = (await db.execute(mq.order_by(MlModel.trained_at.desc()))).scalar_one_or_none()
            if model:
                fq = _sel(Forecast).where(Forecast.model_id == model.id).order_by(Forecast.forecast_date).limit(14)
                if f.org_id is not None:
                    fq = fq.where(Forecast.org_id == f.org_id)
                fc = (await db.execute(fq)).scalars().all()
                if fc:
                    total = sum(float(r.yhat) for r in fc)
                    lines.append(f"- **Forecast (next {len(fc)} days):** {npr(total)} total (~{npr(total/len(fc))}/day, {model.model_type} v{model.version})")
        except Exception:
            pass

        # Catalog hint for discoverability
        try:
            tbl_cnt = (await db.execute(_text("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE' AND table_name NOT LIKE 'alembic%'"))).scalar() or 0
            lines.append(f"\n- **Data catalog:** {tbl_cnt} live tables — ask 'what tables do you have?' or 'sample <table>' to explore any new data instantly")
        except Exception:
            pass

        lines += ["", "**Suggested action:** ask me anything specific — *'revenue yesterday'*, *'how many businesses are registered?'*, *'sample inventory_levels'* — every answer is live, precise, and scoped to you."]
        return "\n".join(lines)
    except Exception:
        logger.warning("_update_handler failed", exc_info=True)
        await _rollback(db)
        return await _generic_fallback(db, f)


async def _generic_fallback(db: AsyncSession, f: Filters) -> str:
    """Minimal live fallback when _update_handler hits an error — never an apology dump."""
    try:
        cards = await _kpi_map(db, f)
        period = _period_label(f)
        rev = cards.get("revenue", {}).get("value")
        exp = cards.get("expense_total", {}).get("value")
        lines = [f"### Live Snapshot — {period}", ""]
        if rev is not None:
            lines.append(f"- **Revenue:** {npr(rev)}")
        if exp is not None:
            lines.append(f"- **Expenses:** {npr(exp)}")
        if not lines[1:]:
            lines.append("- No KPI data loaded yet — connect a data source on the **Data** page")
        lines += ["", "Ask me about **revenue, expenses, products, inventory, anomalies, forecasts, or any table** — I query every table live."]
        return "\n".join(lines)
    except Exception:
        return "I couldn't read live data just now — please try again. The **Data** page shows your sources and freshness."


async def _generic(db: AsyncSession, f: Filters) -> str:
    # Universal: every UNKNOWN phrasing now gets the full live digest, not a 2-line stub
    return await _update_handler(db, f, "")


def _no_data(what: str, period: str | None = None) -> str:
    if period:
        return (
            f"There was no {what} recorded for **{period}** — that period is inside the "
            "loaded data, so this is a genuine zero rather than missing data.\n\n"
            "**Suggested action:** widen the period, or check the **Data** page to confirm "
            "the upload for those days landed."
        )
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
    Intent.BUSINESS: _business,
    Intent.PLATFORM: _platform,
    Intent.USERS: _users_handler,
    Intent.CATALOG: _catalog_handler,
    Intent.UPDATE: _update_handler,
    Intent.GREETING: _greeting,
    Intent.HELP: _help,
    Intent.THANKS: _thanks,
    Intent.CAPABILITIES: _help,
    Intent.UNKNOWN: _update_handler,
}
