"""Scoped read-only query tools for the BI assistant (function-calling).

Every tool is a pure read against the analytics warehouse; none mutate data.
Executors return compact markdown-safe strings the LLM cites directly, so the
assistant answers from *live* numbers instead of the fixed context snapshot.

Executors are role-aware: analysts get the same aggregate views they see in the
dashboard (never line-level money fields).
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.clock import BUSINESS_TZ_NAME, business_today
from app.models import Profile
from app.services.analytics.queries import (
    Filters,
    data_coverage,
    expenses_by_category,
    inventory_levels,
    kpi_timeseries,
    sales_by_dimension,
)

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_DAYS = 30
MAX_ANOMALIES = 50

# Relative expressions the model reaches for constantly. Resolving them here —
# against the business clock — is what makes "yesterday" mean the same thing to
# the assistant as it does to the dashboard.
RELATIVE_DAYS: dict[str, int] = {
    "today": 0,
    "yesterday": 1,
    "day_before_yesterday": 2,
}
RELATIVE_WINDOWS: dict[str, int] = {
    "today": 1,
    "yesterday": 1,
    "last_7_days": 7,
    "last_30_days": 30,
    "last_90_days": 90,
    "last_365_days": 365,
    "this_month": 0,  # handled specially
    "last_month": 0,  # handled specially
}


@dataclass(frozen=True)
class AITool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]  # async (db, user, **kwargs) -> str


def _date() -> dict[str, Any]:
    """Date parameter schema, anchored to the live business date.

    Built per call (not a module constant) so the model always sees the real
    current date in the tool schema instead of a value frozen at import time.
    """
    return {
        "type": "string",
        "description": (
            f"ISO date YYYY-MM-DD in {BUSINESS_TZ_NAME}. Today is "
            f"{business_today().isoformat()}. Omit both dates for the last 30 days."
        ),
    }


def _period() -> dict[str, Any]:
    return {
        "type": "string",
        "enum": sorted(RELATIVE_WINDOWS),
        "description": (
            "Named relative period, resolved on the server against the business "
            "calendar. Use instead of date_from/date_to for phrases like "
            "'yesterday' or 'last month'. Explicit dates win if both are given."
        ),
    }


def _month_start(d: date) -> date:
    return d.replace(day=1)


def _resolve_period(name: str, today: date) -> tuple[date, date] | None:
    if name == "this_month":
        return _month_start(today), today
    if name == "last_month":
        end = _month_start(today) - timedelta(days=1)
        return _month_start(end), end
    if name in RELATIVE_DAYS:
        day = today - timedelta(days=RELATIVE_DAYS[name])
        return day, day
    span = RELATIVE_WINDOWS.get(name)
    if span:
        return today - timedelta(days=span - 1), today
    return None


def _window(kwargs: dict[str, Any]) -> tuple[date, date]:
    """Resolve tool arguments to an inclusive [from, to] business-date window.

    Precedence: explicit date_from/date_to → `date` (single day) → `period`
    → last 30 days. Reversed ranges are swapped rather than returning nothing,
    since models routinely emit them for "from X back to Y" phrasing.
    """
    today = business_today()

    single = _parse_date(kwargs.get("date"), None)
    if single is not None:
        return single, single

    explicit_to = _parse_date(kwargs.get("date_to"), None)
    explicit_from = _parse_date(kwargs.get("date_from"), None)

    if explicit_from is None and explicit_to is None:
        period = kwargs.get("period")
        if period:
            resolved = _resolve_period(str(period), today)
            if resolved:
                return resolved
        return today - timedelta(days=DEFAULT_WINDOW_DAYS - 1), today

    date_to = explicit_to or today
    date_from = explicit_from or (date_to - timedelta(days=DEFAULT_WINDOW_DAYS - 1))
    if date_from > date_to:
        date_from, date_to = date_to, date_from
    return date_from, date_to


def _parse_date(value: Any, default: date | None) -> date | None:
    if isinstance(value, date):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in RELATIVE_DAYS:
        return business_today() - timedelta(days=RELATIVE_DAYS[text])
    try:
        return date.fromisoformat(text)
    except (TypeError, ValueError):
        return default


async def _no_data(db: AsyncSession, date_from: date, date_to: date, what: str) -> str:
    """Explain an empty result honestly.

    "Zero sales on that date" and "we hold no data for that date" are different
    facts, and conflating them is how an assistant ends up reporting a
    confident 0 for a day that was never loaded.
    """
    coverage = await data_coverage(db)
    first, last = coverage.get("first_date"), coverage.get("last_date")
    if first is None or last is None:
        return f"The warehouse has no {what} data loaded at all yet."
    if date_to < first or date_from > last:
        return (
            f"No {what} data exists for {date_from} → {date_to}: the warehouse only "
            f"covers {first} → {last}. This is missing data, not a zero."
        )
    note = ""
    if date_to > last:
        note = (
            f" Note the window runs past the last loaded day ({last}), so any "
            "later days are simply not loaded yet."
        )
    return (
        f"{what.capitalize()} for {date_from} → {date_to} is genuinely zero — the "
        f"window is inside the loaded range ({first} → {last}), there were no "
        f"matching records.{note}"
    )


# ── executors ──────────────────────────────────────────────────────────────

async def _query_kpis(db: AsyncSession, user: Profile, **kwargs: Any) -> str:
    from app.services.analytics.queries import kpi_summary

    date_from, date_to = _window(kwargs)
    cards = await kpi_summary(db, Filters(date_from=date_from, date_to=date_to))
    if not cards or all(c.get("value") in (None, 0) for c in cards):
        return await _no_data(db, date_from, date_to, "KPI")
    lines = [f"KPIs {date_from} → {date_to}:"]
    for c in cards:
        value = c.get("value")
        if value is None:
            continue
        change = c.get("change_pct")
        trend = f" ({change:+.1f}% vs previous period)" if change is not None else ""
        lines.append(f"- {c['metric']}: {value:,.0f}{trend}")
    return "\n".join(lines)


async def _sales_by_dimension(db: AsyncSession, user: Profile, **kwargs: Any) -> str:
    dim = str(kwargs.get("dimension") or "product")
    if dim not in ("product", "category", "channel", "region"):
        return "Unsupported dimension (product|category|channel|region)."
    date_from, date_to = _window(kwargs)
    rows = await sales_by_dimension(db, Filters(date_from=date_from, date_to=date_to), dim)
    if not rows:
        return await _no_data(db, date_from, date_to, f"{dim} sales")
    limit = max(1, min(int(kwargs.get("limit") or 5), 15))
    lines = [f"Top {dim} revenue {date_from} → {date_to}:"]
    for r in rows[:limit]:
        lines.append(
            f"- {r['key']}: {r['revenue']:,.0f} ({r['share_pct']}% share, {r['orders']} orders)"
        )
    return "\n".join(lines)


async def _expenses(db: AsyncSession, user: Profile, **kwargs: Any) -> str:
    date_from, date_to = _window(kwargs)
    rows = await expenses_by_category(db, Filters(date_from=date_from, date_to=date_to))
    if not rows:
        return await _no_data(db, date_from, date_to, "expense")
    limit = max(1, min(int(kwargs.get("limit") or 5), 15))
    lines = [f"Expenses by category {date_from} → {date_to}:"]
    for r in rows[:limit]:
        lines.append(f"- {r['key']}: {r['revenue']:,.0f} ({r['share_pct']}% of expenses)")
    return "\n".join(lines)


async def _timeseries(db: AsyncSession, user: Profile, **kwargs: Any) -> str:
    metric = str(kwargs.get("metric") or "revenue")
    if metric not in ("revenue", "orders", "avg_order_value", "expense_total"):
        return "Unsupported metric."
    date_from, date_to = _window(kwargs)
    granularity = str(kwargs.get("granularity") or "day")
    points = await kpi_timeseries(
        db, Filters(date_from=date_from, date_to=date_to), metric, granularity
    )
    if not points:
        return await _no_data(db, date_from, date_to, metric)
    # Short windows are returned in full: sampling a 7-day question down to a
    # stride would silently drop the very days the user asked about.
    sampled = points if len(points) <= 31 else points[:: max(1, len(points) // 20)][:20]
    lines = [f"{metric} {granularity} series {date_from} → {date_to} ({len(points)} points):"]
    lines += [f"- {p['period']}: {p['value']:,.0f}" for p in sampled]
    if len(sampled) < len(points):
        lines.append(f"(showing {len(sampled)} of {len(points)} points, evenly sampled)")
    return "\n".join(lines)


async def _forecast(db: AsyncSession, user: Profile, **kwargs: Any) -> str:
    from sqlalchemy import select

    from app.models import Forecast, MlModel

    target = str(kwargs.get("target") or "revenue_daily")
    model = (
        await db.execute(
            select(MlModel).where(MlModel.target == target, MlModel.is_active.is_(True))
        )
    ).scalar_one_or_none()
    if model is None:
        return f"No trained forecast model for {target} yet."
    horizon = max(1, min(int(kwargs.get("horizon") or 30), 90))
    rows = (
        (
            await db.execute(
                select(Forecast)
                .where(Forecast.model_id == model.id)
                .order_by(Forecast.forecast_date)
                .limit(horizon)
            )
        )
        .scalars()
        .all()
    )
    if not rows:
        return f"Model for {target} has no projections."
    total = sum(float(r.yhat) for r in rows)
    lo = sum(float(r.yhat_lower) if r.yhat_lower is not None else float(r.yhat) for r in rows)
    hi = sum(float(r.yhat_upper) if r.yhat_upper is not None else float(r.yhat) for r in rows)
    acc = (model.metrics or {}).get("mape")
    acc_txt = f", historical MAPE {acc}%" if acc is not None else ""
    return (
        f"{target} forecast, next {len(rows)} days ({model.model_type} v{model.version}):\n"
        f"- projected total: {total:,.0f}\n"
        f"- confidence band: {lo:,.0f} – {hi:,.0f}\n"
        f"- daily average: {total / len(rows):,.0f}{acc_txt}"
    )


async def _anomalies(db: AsyncSession, user: Profile, **kwargs: Any) -> str:
    from sqlalchemy import select

    from app.models import Anomaly

    stmt = select(Anomaly).order_by(Anomaly.detected_at.desc())
    metric = kwargs.get("metric")
    if metric:
        stmt = stmt.where(Anomaly.metric == str(metric))
    status_filter = kwargs.get("status")
    if status_filter:
        stmt = stmt.where(Anomaly.status == str(status_filter))
    limit = max(1, min(int(kwargs.get("limit") or 10), MAX_ANOMALIES))
    rows = (await db.execute(stmt.limit(limit))).scalars().all()
    if not rows:
        return "No anomalies found."
    lines = [f"Anomalies ({len(rows)}):"]
    for a in rows:
        ctx = a.context or {}
        date_str = ctx.get("date", "?")
        lines.append(
            f"- {a.metric} on {date_str}: observed {float(a.observed_value):,.0f} vs expected "
            f"{float(a.expected_value or 0):,.0f} "
            f"({ctx.get('pct_deviation', '?')}% {ctx.get('direction', '')}, "
            f"severity {a.severity}, status {a.status})"
        )
    return "\n".join(lines)


async def _inventory(db: AsyncSession, user: Profile, **kwargs: Any) -> str:
    below_only = bool(kwargs.get("below_reorder_only", True))
    as_of = _parse_date(kwargs.get("as_of"), None)
    rows = await inventory_levels(db, below_reorder_only=below_only, as_of=as_of)
    if not rows:
        return "No inventory below reorder level." if below_only else "No inventory levels found."
    limit = max(1, min(int(kwargs.get("limit") or 10), 25))
    as_of_txt = f" as of {as_of}" if as_of else ""
    lines = [f"Inventory{as_of_txt} ({len(rows)} item(s) below reorder):"]
    for r in rows[:limit]:
        lines.append(
            f"- {r['product'] or r['sku']}: {r['quantity_on_hand']} on hand, "
            f"reorder at {r['reorder_level']}"
        )
    return "\n".join(lines)


async def _recommendations(db: AsyncSession, user: Profile, **kwargs: Any) -> str:
    from app.services.ml.recommendations import (
        generate_all_recommendations,
        scope_recommendations,
    )

    recs = await scope_recommendations(db, await generate_all_recommendations(db), user)
    limit = max(1, min(int(kwargs.get("limit") or 5), 15))
    if not recs:
        return "No open recommendations right now."
    lines = [f"Top recommendations for you ({len(recs)} available):"]
    for r in recs[:limit]:
        impact = r.get("impact_estimate")
        impact_txt = f" — est. impact {impact:,.0f}" if impact else ""
        lines.append(f"- {r['title']}{impact_txt}")
    return "\n".join(lines)


async def _search_past(db: AsyncSession, user: Profile, **kwargs: Any) -> str:
    from app.services.ai.retrieval import get_retriever

    query_text = str(kwargs.get("query") or "")
    limit = max(1, min(int(kwargs.get("limit") or 5), 10))
    hits = await get_retriever().search(db, query_text, top_k=limit)
    if not hits:
        return "No past insights, anomalies or recommendations match that."
    lines = [f"Past findings close to '{query_text}':"]
    for hit in hits:
        lines.append(f"- [{hit.kind}] {hit.title}: {hit.text[:180]}")
    return "\n".join(lines)


async def _coverage(db: AsyncSession, user: Profile, **kwargs: Any) -> str:
    c = await data_coverage(db)
    if c["first_date"] is None:
        return "The warehouse is empty — no data has been loaded yet."
    lines = [
        f"Today is {c['today']} ({c['timezone']}).",
        f"Warehouse covers {c['first_date']} → {c['last_date']} "
        f"({c['days_behind']} day(s) behind today).",
    ]
    for name in ("sales", "expenses", "inventory"):
        b = c[name]
        if not b["row_count"]:
            lines.append(f"- {name}: no rows loaded")
            continue
        ingested = b["last_ingested_at"]
        stamp = f", last upload {ingested:%Y-%m-%d %H:%M}" if ingested else ""
        lines.append(
            f"- {name}: {b['row_count']:,} rows, {b['first_date']} → {b['last_date']}{stamp}"
        )
    return "\n".join(lines)


# ── registry ───────────────────────────────────────────────────────────────

TOOLS: dict[str, AITool] = {
    "query_kpis": AITool(
        name="query_kpis",
        description=(
            "Headline KPIs (revenue, orders, avg order value, gross margin, expenses) "
            "with % change versus the previous period. Works for a single day "
            "(use `date` or period=today/yesterday) as well as any range."
        ),
        parameters={
            "type": "object",
            "properties": {
                "date_from": _date(),
                "date_to": _date(),
                "date": _date(),
                "period": _period(),
            },
        },
        handler=_query_kpis,
    ),
    "query_sales": AITool(
        name="query_sales",
        description=(
            "Revenue, order counts and share for a dimension (product|category|channel|region) "
            "inside a date window. Use for 'top products' / 'sales by channel' questions."
        ),
        parameters={
            "type": "object",
            "properties": {
                "dimension": {
                    "type": "string",
                    "enum": ["product", "category", "channel", "region"],
                    "description": "What to break sales down by.",
                },
                "date_from": _date(),
                "date_to": _date(),
                "date": _date(),
                "period": _period(),
                "limit": {"type": "integer", "minimum": 1, "maximum": 15, "default": 5},
            },
            "required": ["dimension"],
        },
        handler=_sales_by_dimension,
    ),
    "query_expenses": AITool(
        name="query_expenses",
        description="Expenses by category inside a date window (top categories with share).",
        parameters={
            "type": "object",
            "properties": {
                "date_from": _date(),
                "date_to": _date(),
                "date": _date(),
                "period": _period(),
                "limit": {"type": "integer", "minimum": 1, "maximum": 15, "default": 5},
            },
        },
        handler=_expenses,
    ),
    "query_timeseries": AITool(
        name="query_timeseries",
        description=(
            "Daily/weekly/monthly time series for revenue, orders, avg order value or expenses — "
            "for trend questions."
        ),
        parameters={
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "enum": ["revenue", "orders", "avg_order_value", "expense_total"],
                    "default": "revenue",
                },
                "granularity": {
                    "type": "string",
                    "enum": ["day", "week", "month"],
                    "default": "day",
                },
                "date_from": _date(),
                "date_to": _date(),
                "date": _date(),
                "period": _period(),
            },
        },
        handler=_timeseries,
    ),
    "get_forecast": AITool(
        name="get_forecast",
        description=(
            "Trained model forecast (point + confidence band) for revenue_daily, orders_daily or "
            "expenses_daily. Use for 'what will revenue be' questions."
        ),
        parameters={
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "enum": ["revenue_daily", "orders_daily", "expenses_daily"],
                    "default": "revenue_daily",
                },
                "horizon": {"type": "integer", "minimum": 1, "maximum": 90, "default": 30},
            },
        },
        handler=_forecast,
    ),
    "get_anomalies": AITool(
        name="get_anomalies",
        description=(
            "Latest anomaly-detector alerts (metric, date, observed vs expected, "
            "severity, status). Use for 'anomalies', 'spikes', 'unusual' questions."
        ),
        parameters={
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "enum": ["revenue", "orders", "expense_total"],
                    "description": "Limit to one metric (optional).",
                },
                "status": {
                    "type": "string",
                    "enum": ["open", "acknowledged", "dismissed"],
                    "description": "Optionally limit to a status.",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            },
        },
        handler=_anomalies,
    ),
    "get_inventory": AITool(
        name="get_inventory",
        description="Current inventory levels; SKUs below reorder level by default.",
        parameters={
            "type": "object",
            "properties": {
                "below_reorder_only": {"type": "boolean", "default": True},
                "as_of": _date(),
                "limit": {"type": "integer", "minimum": 1, "maximum": 25, "default": 10},
            },
        },
        handler=_inventory,
    ),
    "get_recommendations": AITool(
        name="get_recommendations",
        description=(
            "Ranked business recommendations with estimated impact (role-scoped). Use for "
            "'what should we do' / 'recommend' questions."
        ),
        parameters={
            "type": "object",
            "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5}},
        },
        handler=_recommendations,
    ),
    "search_past_insights": AITool(
        name="search_past_insights",
        description=(
            "Search past findings: previously generated insights, anomaly explanations and "
            "recommendations. Use for 'have we seen this before?' questions."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Free-text search over past findings."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
            },
            "required": ["query"],
        },
        handler=_search_past,
    ),
    "get_data_coverage": AITool(
        name="get_data_coverage",
        description=(
            "Today's date and exactly which dates the warehouse holds data for, per fact "
            "table, plus when each was last uploaded. CALL THIS FIRST whenever the question "
            "names a specific date, month or 'today'/'yesterday', so you can tell a real "
            "zero apart from a date that was never loaded."
        ),
        parameters={"type": "object", "properties": {}},
        handler=_coverage,
    ),
}


def tool_declarations() -> list[dict[str, Any]]:
    """Provider-agnostic function-tool declarations (Groq/Gemini compatible)."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in TOOLS.values()
    ]


async def dispatch_tool(
    db: AsyncSession, user: Profile, name: str, arguments: dict[str, Any]
) -> str:
    """Execute one tool call inside the current (read-only) request context."""
    tool = TOOLS.get(name)
    if tool is None:
        return f"Unknown tool: {name}"
    try:
        result = await tool.handler(db, user, **arguments)
        return str(result)[:4000]
    except Exception as e:
        logger.exception("tool %s failed", name)
        return f"Tool {name} failed: {e}"