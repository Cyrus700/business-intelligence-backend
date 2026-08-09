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

from app.models import Profile
from app.services.analytics.queries import (
    Filters,
    expenses_by_category,
    inventory_levels,
    kpi_timeseries,
    sales_by_dimension,
)

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_DAYS = 30
MAX_ANOMALIES = 50
WINDOW_HELP = "ISO date YYYY-MM-DD. Default: last 30 days ending today."


@dataclass(frozen=True)
class AITool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]  # async (db, user, **kwargs) -> str


def _date() -> dict[str, Any]:
    return {
        "type": "string",
        "description": "ISO date YYYY-MM-DD. Default: last 30 days ending today.",
    }


def _window(kwargs: dict[str, Any]) -> tuple[date, date]:
    today = date.today()
    date_to = _parse_date(kwargs.get("date_to"), today)
    date_from = _parse_date(
        kwargs.get("date_from"), date_to - timedelta(days=DEFAULT_WINDOW_DAYS - 1)
    )
    return date_from, date_to


def _parse_date(value: Any, default: date) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return default


# ── executors ──────────────────────────────────────────────────────────────

async def _query_kpis(db: AsyncSession, user: Profile, **kwargs: Any) -> str:
    from app.services.analytics.queries import kpi_summary

    date_from, date_to = _window(kwargs)
    cards = await kpi_summary(db, Filters(date_from=date_from, date_to=date_to))
    if not cards:
        return "No KPI data in that window."
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
        return f"No {dim} sales in that window."
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
        return "No expense data in that window."
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
        return f"No {metric} time-series in that window."
    sampled = points[:: max(1, len(points) // 20)][:20]
    lines = [f"{metric} {granularity} series {date_from} → {date_to}:"]
    lines += [f"- {p['period']}: {p['value']:,.0f}" for p in sampled]
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
    rows = await inventory_levels(db, below_reorder_only=below_only)
    if not rows:
        return "No inventory below reorder level." if below_only else "No inventory levels found."
    limit = max(1, min(int(kwargs.get("limit") or 10), 25))
    lines = [f"Inventory ({len(rows)} item(s) below reorder):"]
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


# ── registry ───────────────────────────────────────────────────────────────

TOOLS: dict[str, AITool] = {
    "query_kpis": AITool(
        name="query_kpis",
        description=(
            "Headline KPIs (revenue, orders, avg order value, gross margin, expenses) "
            "with % change versus the previous period, inside an optional date window."
        ),
        parameters={
            "type": "object",
            "properties": {
                "date_from": _date(),
                "date_to": _date(),
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