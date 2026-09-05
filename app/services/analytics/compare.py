"""Period-over-period comparison service — compare 2..N months or years side-by-side.

Powers the business Compare view (professional, multi-period analysis with
illustrations and AI suggestions). Every query is org-scoped, permission-aware,
and built on the same fact-table aggregates that drive the rest of the
dashboard — no new tables, no invented numbers.

Design
------
* Accepts 2–6 arbitrary periods (date_from → date_to, label optional). The
  frontend's Month / Year pickers produce these, but custom ranges work too.
* For each period the service gathers:
  - headline KPIs (revenue, orders, AOV, gross margin, expense, net)
  - dimensional breakdowns (category, channel, region, product)
  - optional daily/weekly/monthly timeseries for an overlay chart
* All heavy aggregation is parallelized with asyncio.gather and reuses
  ``queries._sales_kpis`` / ``_expense_kpi`` / ``sales_by_dimension``.
* Per-day normalization is computed alongside totals so months of different
  lengths are comparable without bias.
* Deterministic insights (highlights, drivers, watch-outs) are derived arith-
  metically so the AI layer can only add narrative, never numbers.
* AI suggestions are generated on demand via the provider fallback chain
  (Groq → Gemini → local) and are always marked as AI-generated.
* In-memory TTL cache (60s) prevents repeated identical compares from
  hammering the warehouse.

All heavy aggregation reuses ``queries._sales_kpis`` / ``_expense_kpi`` /
``sales_by_dimension`` so planner indexes and cache behaviour stay identical.
"""

from __future__ import annotations

import asyncio
import calendar
import hashlib
import json
import logging
import math
import time
from datetime import date, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.clock import business_today
from app.services.analytics.queries import (
    Filters,
    _expense_kpi,
    _sales_kpis,
    expenses_by_category,
    kpi_timeseries,
    sales_by_dimension,
)

logger = logging.getLogger(__name__)

ALLOWED_METRICS = ("revenue", "orders", "avg_order_value", "gross_margin", "expense_total", "net_profit")
ALLOWED_DIMS = ("category", "channel", "region", "product", "expense_category")
MAX_PERIODS = 6
MIN_PERIODS = 2
MAX_PERIOD_SPAN_DAYS = 400  # ~13 months; yearly compare fits, absurd windows don't
CACHE_TTL_S = 60

# Label helpers
_METRIC_LABELS: dict[str, str] = {
    "revenue": "Revenue",
    "orders": "Orders",
    "avg_order_value": "Avg Order Value",
    "gross_margin": "Gross Margin",
    "expense_total": "Expenses",
    "net_profit": "Net Profit",
}
_METRIC_UNITS: dict[str, str] = {
    "revenue": "NPR",
    "orders": "orders",
    "avg_order_value": "NPR",
    "gross_margin": "NPR",
    "expense_total": "NPR",
    "net_profit": "NPR",
}

# ── Simple TTL cache for compare results ───────────────────────────────
_compare_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def _cache_key(
    org_id: UUID | None,
    periods: list[dict],
    metrics: list[str],
    dims: list[str],
    inc_ts: bool,
    ts_metric: str,
    ts_gran: str,
    can_pnl: bool,
) -> str:
    payload = json.dumps(
        {
            "org": str(org_id) if org_id else None,
            "periods": [(str(p["from"]), str(p["to"])) for p in periods],
            "metrics": sorted(metrics),
            "dims": sorted(dims),
            "ts": f"{inc_ts}:{ts_metric}:{ts_gran}:{can_pnl}",
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_get(key: str) -> dict[str, Any] | None:
    hit = _compare_cache.get(key)
    if not hit:
        return None
    exp, val = hit
    if time.time() > exp:
        _compare_cache.pop(key, None)
        return None
    return val


def _cache_set(key: str, val: dict[str, Any]) -> None:
    # Cap size
    if len(_compare_cache) > 128:
        # evict oldest
        oldest = min(_compare_cache.items(), key=lambda kv: kv[1][0])[0]
        _compare_cache.pop(oldest, None)
    _compare_cache[key] = (time.time() + CACHE_TTL_S, val)


def month_bounds(year: int, month: int) -> tuple[date, date]:
    last = calendar.monthrange(year, month)[1]
    return date(year, month, 1), date(year, month, last)


def year_bounds(year: int) -> tuple[date, date]:
    return date(year, 1, 1), date(year, 12, 31)


def quarter_bounds(year: int, quarter: int) -> tuple[date, date]:
    """1-indexed quarter bounds."""
    start_month = (quarter - 1) * 3 + 1
    end_month = start_month + 2
    last = calendar.monthrange(year, end_month)[1]
    return date(year, start_month, 1), date(year, end_month, last)


def _validate_periods(periods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not (MIN_PERIODS <= len(periods) <= MAX_PERIODS):
        raise ValueError(f"Provide {MIN_PERIODS}–{MAX_PERIODS} periods (got {len(periods)})")
    out: list[dict[str, Any]] = []
    seen: set[tuple[date, date]] = set()
    for idx, p in enumerate(periods):
        raw_from = p.get("from") or p.get("date_from") or p.get("start")
        raw_to = p.get("to") or p.get("date_to") or p.get("end")
        if not raw_from or not raw_to:
            raise ValueError(f"Period {idx + 1}: 'from' and 'to' are required (YYYY-MM-DD)")
        try:
            d_from = date.fromisoformat(str(raw_from)[:10])
            d_to = date.fromisoformat(str(raw_to)[:10])
        except Exception:
            raise ValueError(f"Period {idx + 1}: invalid date format (expected YYYY-MM-DD)") from None
        if d_from > d_to:
            d_from, d_to = d_to, d_from
        span = (d_to - d_from).days + 1
        if span <= 0:
            raise ValueError(f"Period {idx + 1}: span must be >=1 day")
        if span > MAX_PERIOD_SPAN_DAYS:
            raise ValueError(f"Period {idx + 1}: span {span} days exceeds {MAX_PERIOD_SPAN_DAYS}-day limit")
        if d_to > business_today() + timedelta(days=1):
            raise ValueError(f"Period {idx + 1}: ends in the future ({d_to})")
        if d_from < date(2000, 1, 1):
            raise ValueError(f"Period {idx + 1}: start before 2000 not allowed")
        key = (d_from, d_to)
        if key in seen:
            raise ValueError(f"Period {idx + 1}: duplicate range {d_from} → {d_to}")
        seen.add(key)
        label = p.get("label") or f"{d_from.isoformat()} → {d_to.isoformat()}"
        # Use month label when period is exactly one calendar month
        if span <= 31:
            try:
                if (
                    d_from.day == 1
                    and d_to.day == calendar.monthrange(d_to.year, d_to.month)[1]
                    and d_from.month == d_to.month
                ):
                    label = p.get("label") or d_from.strftime("%B %Y")
            except Exception:
                pass
        # Use quarter label when exactly one quarter
        if span >= 89 and span <= 92:
            try:
                for q in (1, 2, 3, 4):
                    qb_s, qb_e = quarter_bounds(d_from.year, q)
                    if d_from == qb_s and d_to == qb_e:
                        label = p.get("label") or f"Q{q} {d_from.year}"
                        break
            except Exception:
                pass
        # Use year label when period is exactly one calendar year
        if d_from.month == 1 and d_from.day == 1 and d_to.month == 12 and d_to.day == 31 and d_from.year == d_to.year:
            label = p.get("label") or str(d_from.year)
        out.append({"id": p.get("id") or f"p{idx + 1}", "label": label, "from": d_from, "to": d_to})
    # Detect overlapping periods — warn but allow (comparison still valid, just note)
    out_sorted = sorted(out, key=lambda x: x["from"])
    overlaps: list[str] = []
    for i in range(len(out_sorted) - 1):
        if out_sorted[i]["to"] >= out_sorted[i + 1]["from"]:
            overlaps.append(f"{out_sorted[i]['label']} overlaps {out_sorted[i + 1]['label']}")
    if overlaps:
        logger.info("compare overlapping periods: %s", "; ".join(overlaps))
    # sort chronologically for stable charts
    out.sort(key=lambda x: x["from"])
    # re-id after sort to keep sequential
    for i, o in enumerate(out):
        o["id"] = f"p{i + 1}"
    # attach overlap warning to first item for payload
    if overlaps:
        out[0]["_overlap_warning"] = "; ".join(overlaps)
    return out


async def _period_metrics(
    db: AsyncSession,
    d_from: date,
    d_to: date,
    org_id: UUID | None,
    *,
    include_net: bool = True,
) -> dict[str, float]:
    """KPIs for a single window — mirrors ``kpi_summary`` but for one period."""
    f = Filters(date_from=d_from, date_to=d_to, org_id=org_id)
    sales, expense = await asyncio.gather(
        _sales_kpis(db, f, d_from, d_to),
        _expense_kpi(db, d_from, d_to, org_id),
    )
    gross = float(sales.get("gross_margin", 0.0))
    revenue = float(sales.get("revenue", 0.0))
    orders = float(sales.get("orders", 0.0))
    aov = float(sales.get("avg_order_value", 0.0))
    expense_f = float(expense)
    net = gross - expense_f
    span = (d_to - d_from).days + 1
    out: dict[str, float] = {
        "revenue": round(revenue, 2),
        "orders": round(orders, 2),
        "avg_order_value": round(aov, 2),
        "gross_margin": round(gross, 2),
        "expense_total": round(expense_f, 2),
        # per-day normalizations for fair comparison across different lengths
        "revenue_per_day": round(revenue / span, 2) if span else 0,
        "orders_per_day": round(orders / span, 2) if span else 0,
        "expense_per_day": round(expense_f / span, 2) if span else 0,
    }
    if include_net:
        out["net_profit"] = round(net, 2)
        out["net_per_day"] = round(net / span, 2) if span else 0
    return out


async def _period_dimension(
    db: AsyncSession,
    d_from: date,
    d_to: date,
    org_id: UUID | None,
    dimension: str,
) -> list[dict[str, Any]]:
    if dimension == "expense_category":
        rows = await expenses_by_category(db, Filters(date_from=d_from, date_to=d_to, org_id=org_id))
        return [
            {"key": r["key"], "revenue": r["revenue"], "orders": r["orders"], "share_pct": r["share_pct"]} for r in rows
        ]
    dim = dimension
    if dim not in ("product", "category", "channel", "region"):
        raise ValueError(f"Unsupported dimension: {dimension}")
    rows = await sales_by_dimension(db, Filters(date_from=d_from, date_to=d_to, org_id=org_id), dim)
    return rows


def _delta(a: float, b: float) -> tuple[float, float | None]:
    """(absolute delta, percent change from b to a). b is baseline."""
    delta = a - b
    pct = (delta / b * 100) if b else None
    return delta, pct


def _stddev(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals))


def _cagr(start: float, end: float, periods: int) -> float | None:
    if start <= 0 or periods <= 1:
        return None
    try:
        return (pow(end / start, 1 / (periods - 1)) - 1) * 100
    except Exception:
        return None


async def compare_periods(
    db: AsyncSession,
    *,
    periods: list[dict[str, Any]],
    metrics: list[str] | None = None,
    dimensions: list[str] | None = None,
    org_id: UUID | None = None,
    include_timeseries: bool = True,
    timeseries_metric: str = "revenue",
    timeseries_granularity: str = "day",
    can_view_pnl: bool = True,
) -> dict[str, Any]:
    """Core comparison — returns the payload the API serialises.

    ``periods`` is a list of ``{from, to, label?}`` with ISO dates.
    ``metrics`` and ``dimensions`` are filtered to the allow-list.
    """
    metrics = [
        m
        for m in (metrics or ["revenue", "orders", "avg_order_value", "gross_margin", "expense_total"])
        if m in ALLOWED_METRICS
    ]
    if not can_view_pnl:
        metrics = [m for m in metrics if m not in ("expense_total", "net_profit", "gross_margin")]
        if not metrics:
            metrics = ["revenue", "orders"]
    if timeseries_metric not in ALLOWED_METRICS:
        timeseries_metric = "revenue"

    dimensions = [d for d in (dimensions or ["category", "channel", "region"]) if d in ALLOWED_DIMS]
    if not can_view_pnl:
        dimensions = [d for d in dimensions if d != "expense_category"]
    if len(dimensions) > 4:
        dimensions = dimensions[:4]

    validated = _validate_periods(periods)
    n = len(validated)

    # Cache lookup
    cache_key = _cache_key(
        org_id,
        validated,
        metrics,
        dimensions,
        include_timeseries,
        timeseries_metric,
        timeseries_granularity,
        can_view_pnl,
    )
    cached = _cache_get(cache_key)
    if cached is not None:
        # Return shallow copy with fresh generated_at
        out = dict(cached)
        out["meta"] = dict(cached["meta"])
        out["meta"]["cached"] = True
        return out

    # ── KPIs per period (parallel) ─────────────────────────────────────
    period_vals_list = await asyncio.gather(
        *[_period_metrics(db, p["from"], p["to"], org_id, include_net=can_view_pnl) for p in validated]
    )
    period_payloads: list[dict[str, Any]] = []
    for p, vals in zip(validated, period_vals_list):
        filtered_vals = {
            k: v
            for k, v in vals.items()
            if k in metrics or k in ("revenue_per_day", "orders_per_day", "expense_per_day", "net_per_day")
        }
        # Always include per-day for the metrics that are selected
        period_payloads.append(
            {
                "id": p["id"],
                "label": p["label"],
                "from": p["from"].isoformat(),
                "to": p["to"].isoformat(),
                "span_days": (p["to"] - p["from"]).days + 1,
                "metrics": filtered_vals,
                "_full_metrics": vals,
            }
        )

    # ── KPI comparison matrix ──────────────────────────────────────────
    kpi_comparison: list[dict[str, Any]] = []
    for metric in metrics:
        values = [float(pp["_full_metrics"].get(metric, 0.0)) for pp in period_payloads]
        # per-day values for fair length comparison
        per_day_vals = [
            float(pp["_full_metrics"].get(f"{metric}_per_day", pp["_full_metrics"].get(metric, 0.0) / pp["span_days"]))
            if pp["span_days"]
            else 0
            for pp in period_payloads
        ]
        # deltas vs first period and vs previous
        deltas_vs_first: list[float | None] = []
        pct_vs_first: list[float | None] = []
        deltas_prev: list[float | None] = []
        pct_prev: list[float | None] = []
        per_day_pct_first: list[float | None] = []
        for i, v in enumerate(values):
            if i == 0:
                deltas_vs_first.append(None)
                pct_vs_first.append(None)
                deltas_prev.append(None)
                pct_prev.append(None)
                per_day_pct_first.append(None)
            else:
                d_first, p_first = _delta(v, values[0])
                d_prev, p_prev = _delta(v, values[i - 1])
                _, pd_first = _delta(per_day_vals[i], per_day_vals[0])
                deltas_vs_first.append(round(d_first, 2))
                pct_vs_first.append(round(p_first, 1) if p_first is not None else None)
                deltas_prev.append(round(d_prev, 2))
                pct_prev.append(round(p_prev, 1) if p_prev is not None else None)
                per_day_pct_first.append(round(pd_first, 1) if pd_first is not None else None)
        total_change = (values[-1] - values[0]) if n >= 2 else 0.0
        total_pct = (total_change / values[0] * 100) if values[0] else None
        # advanced stats
        avg = sum(values) / len(values) if values else 0
        sd = _stddev(values)
        cv = (sd / avg * 100) if avg else None
        cagr = _cagr(values[0], values[-1], n)
        # best/worst period
        best_idx = max(range(len(values)), key=lambda i: values[i]) if values else 0
        worst_idx = min(range(len(values)), key=lambda i: values[i]) if values else 0
        kpi_comparison.append(
            {
                "metric": metric,
                "label": _METRIC_LABELS.get(metric, metric),
                "unit": _METRIC_UNITS.get(metric, ""),
                "values": values,
                "per_day_values": [round(v, 2) for v in per_day_vals],
                "deltas_vs_first": deltas_vs_first,
                "pct_vs_first": pct_vs_first,
                "deltas_vs_prev": deltas_prev,
                "pct_vs_prev": pct_prev,
                "per_day_pct_vs_first": per_day_pct_first,
                "total_delta": round(total_change, 2),
                "total_pct": round(total_pct, 1) if total_pct is not None else None,
                "min": round(min(values), 2) if values else 0,
                "max": round(max(values), 2) if values else 0,
                "avg": round(avg, 2),
                "stddev": round(sd, 2),
                "cv_pct": round(cv, 1) if cv is not None else None,
                "cagr_pct": round(cagr, 1) if cagr is not None else None,
                "best_period": period_payloads[best_idx]["label"] if period_payloads else None,
                "worst_period": period_payloads[worst_idx]["label"] if period_payloads else None,
                "trend": "up" if total_change > 0 else ("down" if total_change < 0 else "flat"),
            }
        )

    # ── Dimensional breakdowns (parallel per dimension) ────────────────
    async def _build_dim(dim: str) -> tuple[str, dict[str, Any]]:
        per_period_rows = await asyncio.gather(
            *[_period_dimension(db, p["from"], p["to"], org_id, dim) for p in validated]
        )
        all_keys: set[str] = set()
        for rows in per_period_rows:
            for r in rows:
                all_keys.add(r["key"])
        series: list[dict[str, Any]] = []
        for key in sorted(all_keys):
            vals: list[float] = []
            shares: list[float] = []
            orders_list: list[int] = []
            per_day: list[float] = []
            for idx, rows in enumerate(per_period_rows):
                hit = next((r for r in rows if r["key"] == key), None)
                v = float(hit["revenue"]) if hit else 0.0
                vals.append(v)
                shares.append(float(hit["share_pct"]) if hit and "share_pct" in hit else 0.0)
                orders_list.append(int(hit["orders"]) if hit and "orders" in hit else 0)
                span = validated[idx]["to"] - validated[idx]["from"]
                days = span.days + 1
                per_day.append(round(v / days, 2) if days else 0)
            d_first = vals[-1] - vals[0] if n >= 2 else 0.0
            pct_first = (d_first / vals[0] * 100) if vals[0] else None
            change_direction = "up" if d_first > 0 else ("down" if d_first < 0 else "flat")
            mean_val = sum(vals) / len(vals) if vals else 0
            sd = _stddev(vals)
            cv = (sd / mean_val * 100) if mean_val else None
            is_growing = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1)) if len(vals) > 1 else False
            is_declining = all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)) if len(vals) > 1 else False
            trend = (
                "growing"
                if is_growing and d_first != 0
                else ("declining" if is_declining and d_first != 0 else "mixed")
            )
            series.append(
                {
                    "key": key,
                    "values": [round(v, 2) for v in vals],
                    "per_day": per_day,
                    "shares": [round(s, 1) for s in shares],
                    "orders": orders_list,
                    "delta_vs_first": round(d_first, 2),
                    "pct_vs_first": round(pct_first, 1) if pct_first is not None else None,
                    "direction": change_direction,
                    "trend": trend,
                    "total": round(sum(vals), 2),
                    "avg": round(mean_val, 2),
                    "stddev": round(sd, 2),
                    "cv_pct": round(cv, 1) if cv is not None else None,
                }
            )
        series.sort(key=lambda x: x["total"], reverse=True)
        top = series[:12]
        if len(series) > 12:
            rest_vals = [0.0] * n
            rest_per_day = [0.0] * n
            for s in series[12:]:
                for i, v in enumerate(s["values"]):
                    rest_vals[i] += v
                    rest_per_day[i] += s["per_day"][i]
            other = {
                "key": "Other",
                "values": [round(v, 2) for v in rest_vals],
                "per_day": [round(v, 2) for v in rest_per_day],
                "shares": [round(sum(s["shares"][i] for s in series[12:]), 1) for i in range(n)],
                "orders": [sum(s["orders"][i] for s in series[12:]) for i in range(n)],
                "delta_vs_first": round(rest_vals[-1] - rest_vals[0], 2) if n >= 2 else 0,
                "pct_vs_first": round((rest_vals[-1] - rest_vals[0]) / rest_vals[0] * 100, 1)
                if n >= 2 and rest_vals[0]
                else None,
                "direction": "up"
                if rest_vals[-1] > rest_vals[0]
                else ("down" if rest_vals[-1] < rest_vals[0] else "flat"),
                "trend": "mixed",
                "total": round(sum(rest_vals), 2),
                "avg": round(sum(rest_vals) / n, 2) if n else 0,
                "stddev": round(_stddev(rest_vals), 2),
                "cv_pct": None,
            }
            top.append(other)
        totals_per_period = [sum(r["revenue"] for r in rows) for rows in per_period_rows]
        per_day_totals = [
            round(
                t / validated[i]["to"].day if False else t / ((validated[i]["to"] - validated[i]["from"]).days + 1), 2
            )
            for i, t in enumerate(totals_per_period)
        ]
        # actually per_day_totals correct calc:
        per_day_totals = [
            round(t / ((validated[i]["to"] - validated[i]["from"]).days + 1), 2)
            for i, t in enumerate(totals_per_period)
        ]
        return dim, {
            "dimension": dim,
            "period_labels": [p["label"] for p in validated],
            "totals": [round(t, 2) for t in totals_per_period],
            "per_day_totals": per_day_totals,
            "series": top,
            "all_keys_count": len(series),
            "top_gainer": max(top, key=lambda x: x["delta_vs_first"]) if top else None,
            "top_decliner": min(top, key=lambda x: x["delta_vs_first"]) if top else None,
        }

    dimensional: dict[str, Any] = {}
    if dimensions:
        dim_results = await asyncio.gather(*[_build_dim(d) for d in dimensions])
        dimensional = {k: v for k, v in dim_results}

    # ── Timeseries overlay ───────────────────────────────────────────
    timeseries_overlay: dict[str, Any] | None = None
    if include_timeseries:
        gran = timeseries_granularity
        max_span = max((p["to"] - p["from"]).days for p in validated)
        if gran == "day" and max_span > 90:
            gran = "month" if max_span > 180 else "week"

        async def _fetch_ts(p):
            f = Filters(date_from=p["from"], date_to=p["to"], org_id=org_id)
            try:
                pts = await kpi_timeseries(db, f, timeseries_metric, gran)  # type: ignore[arg-type]
            except Exception as e:
                logger.warning("timeseries failed for %s: %s", p["label"], e)
                pts = []
            return {
                "period_id": p["id"],
                "period_label": p["label"],
                "granularity": gran,
                "points": [{"period": str(pt["period"]), "value": pt["value"]} for pt in pts],
            }

        overlay_series = await asyncio.gather(*[_fetch_ts(p) for p in validated])
        timeseries_overlay = {
            "metric": timeseries_metric,
            "granularity": gran,
            "series": list(overlay_series),
        }

    # ── Deterministic insights ───────────────────────────────────────
    insights = _build_insights(period_payloads, kpi_comparison, dimensional, validated)

    # strip internal _full_metrics before returning
    for pp in period_payloads:
        pp.pop("_full_metrics", None)

    # overlap warning passthrough
    overlap_warning = validated[0].get("_overlap_warning") if validated and "_overlap_warning" in validated[0] else None

    result = {
        "periods": period_payloads,
        "kpi_comparison": kpi_comparison,
        "dimensional": dimensional,
        "timeseries_overlay": timeseries_overlay,
        "insights": insights,
        "warnings": [overlap_warning] if overlap_warning else [],
        "meta": {
            "generated_at": business_today().isoformat(),
            "timezone": "Asia/Kathmandu",
            "org_scoped": org_id is not None,
            "metrics": metrics,
            "dimensions": dimensions,
            "periods_count": n,
            "cached": False,
        },
    }
    _cache_set(cache_key, result)
    return result


def _build_insights(
    period_payloads: list[dict[str, Any]],
    kpi_comparison: list[dict[str, Any]],
    dimensional: dict[str, Any],
    validated: list[dict[str, Any]],
) -> dict[str, Any]:
    """Pure-arithmetic narrative pieces — no LLM, no hallucination."""
    labels = [p["label"] for p in period_payloads]

    highlights: list[str] = []
    drivers: list[str] = []
    watchouts: list[str] = []
    stats: dict[str, Any] = {}
    momentum: list[str] = []

    # KPI highlights
    for kc in kpi_comparison:
        metric = kc["metric"]
        total_pct = kc["total_pct"]
        total_delta = kc["total_delta"]
        if total_pct is None:
            continue
        label = kc["label"]
        first_label = labels[0]
        last_label = labels[-1]
        direction = "grew" if total_delta > 0 else ("fell" if total_delta < 0 else "was flat")
        if abs(total_pct) >= 3:
            highlights.append(
                f"{label} {direction} {abs(total_pct):.1f}% from {first_label} to {last_label} "
                f"({kc['values'][0]:,.0f} → {kc['values'][-1]:,.0f}, Δ {total_delta:+,.0f})"
            )
            # per-day context if periods differ in length
            if kc.get("per_day_values"):
                pd_vals = kc["per_day_values"]
                if pd_vals[0] and pd_vals[-1]:
                    pd_pct = (pd_vals[-1] - pd_vals[0]) / pd_vals[0] * 100
                    if abs(pd_pct - total_pct) > 5:
                        stats[f"{metric}_per_day"] = (
                            f"Per-day {pd_vals[0]:,.0f} → {pd_vals[-1]:,.0f} ({pd_pct:+.1f}%) — length-adjusted"
                        )
        # volatility insight
        vals = kc["values"]
        if len(vals) >= 3:
            up = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
            down = all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
            if up and total_pct and abs(total_pct) >= 5:
                highlights.append(
                    f"{label} rose every period — consistent growth from {first_label} through {last_label}."
                )
                momentum.append(f"{label} momentum intact ({len(vals)} consecutive gains)")
            elif down and total_pct and abs(total_pct) >= 5:
                watchouts.append(f"{label} fell every period — steady decline that needs attention.")
            elif not (up or down):
                peak = max(vals)
                trough = min(vals)
                if peak and trough and (peak - trough) / peak > 0.15:
                    stats[f"{metric}_volatility"] = (
                        f"Volatile: range {trough:,.0f} – {peak:,.0f} ({(peak - trough) / peak * 100:.0f}% swing, CV {kc.get('cv_pct', '?')}%)"
                    )
                # best vs worst
                best = kc.get("best_period")
                worst = kc.get("worst_period")
                if best and worst and best != worst:
                    stats[f"{metric}_best_worst"] = f"Best {best} vs worst {worst}"

        # CAGR for 3+ periods
        if kc.get("cagr_pct") is not None:
            stats[f"{metric}_cagr"] = f"CAGR {kc['cagr_pct']:+.1f}% across {len(vals)} periods"

    if not highlights:
        highlights.append(f"Periods {', '.join(labels)} are within 3% on headline KPIs — broadly stable.")

    # Dimensional drivers / drags
    for dim, payload in dimensional.items():
        series = payload.get("series", [])
        top_gainer = payload.get("top_gainer")
        top_decliner = payload.get("top_decliner")
        if top_gainer and top_gainer["delta_vs_first"] and abs(top_gainer["delta_vs_first"]) > 0:
            if top_gainer["pct_vs_first"] is not None and abs(top_gainer["pct_vs_first"]) >= 10:
                drivers.append(
                    f"{dim.title()} '{top_gainer['key']}' drove growth: {top_gainer['values'][0]:,.0f} → {top_gainer['values'][-1]:,.0f} "
                    f"({top_gainer['pct_vs_first']:+.1f}%, {top_gainer['trend']})"
                )
        if top_decliner and top_decliner["delta_vs_first"] and top_decliner["delta_vs_first"] < 0:
            if top_decliner["pct_vs_first"] is not None and top_decliner["pct_vs_first"] <= -10:
                watchouts.append(
                    f"{dim.title()} '{top_decliner['key']}' pulled back: {top_decliner['values'][0]:,.0f} → {top_decliner['values'][-1]:,.0f} "
                    f"({top_decliner['pct_vs_first']:.1f}%)"
                )
        # concentration shift
        if len(series) >= 2:
            leader = series[0]
            if len(leader["shares"]) >= 2:
                share_delta = leader["shares"][-1] - leader["shares"][0]
                if abs(share_delta) >= 5:
                    watchouts.append(
                        f"Concentration shift in {dim}: '{leader['key']}' share moved {share_delta:+.1f} pp "
                        f"({leader['shares'][0]:.1f}% → {leader['shares'][-1]:.1f}%)"
                    )
                for idx, period_label in enumerate(labels):
                    shares = [s["shares"][idx] for s in series if idx < len(s["shares"])]
                    hhi = sum((sh / 100) ** 2 for sh in shares if sh)
                    if idx == 0:
                        first_hhi = hhi
                    if idx == len(labels) - 1:
                        last_hhi = hhi
                if "first_hhi" in locals() and "last_hhi" in locals():
                    if last_hhi - first_hhi > 0.05:
                        watchouts.append(
                            f"Revenue is more concentrated in {dim} in {labels[-1]} than in {labels[0]} (HHI {first_hhi:.2f} → {last_hhi:.2f})"
                        )
                    elif first_hhi - last_hhi > 0.05:
                        drivers.append(
                            f"Revenue spread is healthier in {dim} by {labels[-1]} (HHI {first_hhi:.2f} → {last_hhi:.2f})"
                        )
        # high CV members
        high_cv = [s for s in series if s.get("cv_pct") is not None and s["cv_pct"] > 40]
        if high_cv:
            most_volatile = max(high_cv, key=lambda x: x["cv_pct"])
            stats[f"{dim}_most_volatile"] = (
                f"Most volatile {dim}: '{most_volatile['key']}' CV {most_volatile['cv_pct']}%"
            )

    # overall verdict
    revenue_kc = next((k for k in kpi_comparison if k["metric"] == "revenue"), None)
    if revenue_kc and revenue_kc["total_pct"] is not None:
        if revenue_kc["total_pct"] >= 10:
            verdict = "Strong growth — headline revenue is up materially."
        elif revenue_kc["total_pct"] >= 3:
            verdict = "Modest growth — revenue is up slightly."
        elif revenue_kc["total_pct"] <= -10:
            verdict = "Material decline — revenue is down sharply."
        elif revenue_kc["total_pct"] <= -3:
            verdict = "Soft — revenue is down slightly."
        else:
            verdict = "Stable — headline revenue is essentially flat."
    else:
        verdict = "Comparison complete."

    return {
        "verdict": verdict,
        "highlights": highlights[:6],
        "drivers": drivers[:5],
        "watchouts": watchouts[:5],
        "momentum": momentum[:3],
        "stats": stats,
        "method": "Arithmetic over warehouse aggregates — deterministic, no LLM.",
    }


# ── AI suggestions ──────────────────────────────────────────────────────────


async def generate_ai_suggestions(
    *,
    comparison: dict[str, Any],
    role: str = "analyst",
    org_name: str | None = None,
) -> dict[str, Any]:
    """LLM narrative for a comparison.

    Uses the provider chain (Groq → Gemini) with a deterministic fallback that
    simply formats the deterministic insights when no provider is configured or
    every provider fails. The caller should already have the deterministic
    ``comparison`` payload from :func:`compare_periods`.
    """
    periods = comparison.get("periods", [])
    kpi_comparison = comparison.get("kpi_comparison", [])
    insights = comparison.get("insights", {})
    dimensional = comparison.get("dimensional", {})

    period_lines = [
        f"- {p['label']}: {p['from']} → {p['to']} ({p['span_days']} days, per-day revenue {p['metrics'].get('revenue_per_day', 'n/a')})"
        for p in periods
    ]
    kpi_lines: list[str] = []
    for kc in kpi_comparison:
        vals = " → ".join(f"{v:,.0f}" for v in kc["values"])
        pct = kc["total_pct"]
        pct_txt = f"{pct:+.1f}%" if pct is not None else "n/a"
        pd_vals = kc.get("per_day_values") or []
        pd_txt = ""
        if pd_vals:
            pd_str = " → ".join(f"{v:,.0f}/day" for v in pd_vals)
            pd_txt = f" | per-day: {pd_str}"
        kpi_lines.append(
            f"- {kc['label']}: {vals} (Δ {kc['total_delta']:+,.0f}, {pct_txt}){pd_txt} CAGR {kc.get('cagr_pct', 'n/a')}%"
        )

    dim_lines: list[str] = []
    for dim, payload in dimensional.items():
        top = payload.get("series", [])[:3]
        if not top:
            continue
        items = ", ".join(
            f"{s['key']} {s['values'][0]:,.0f}→{s['values'][-1]:,.0f} ({s['pct_vs_first'] if s['pct_vs_first'] is not None else 'n/a'}%, CV {s.get('cv_pct', '?')}%)"
            for s in top
        )
        dim_lines.append(f"- {dim}: {items}")

    deterministic_bullets = "\n".join(
        [
            f"Verdict: {insights.get('verdict', '')}",
            f"Highlights: {'; '.join(insights.get('highlights', [])[:3])}",
            f"Drivers: {'; '.join(insights.get('drivers', [])[:2])}",
            f"Watch-outs: {'; '.join(insights.get('watchouts', [])[:2])}",
            f"Momentum: {'; '.join(insights.get('momentum', [])[:2])}",
        ]
    )

    prompt = f"""You are a senior BI analyst writing for a {role} in {org_name or "the business"}.

Compare these periods head-to-head and give concise, actionable takeaways.

PERIODS (note span_days and per-day normalization for fair comparison):
{chr(10).join(period_lines)}

KPIS (includes per-day and CAGR for multi-period trends):
{chr(10).join(kpi_lines)}

TOP DIMENSIONAL MOVERS (CV% = volatility):
{chr(10).join(dim_lines) if dim_lines else "- (no dimensional breakdown requested)"}

DETERMINISTIC FINDINGS (ground truth — do not contradict):
{deterministic_bullets}

Write the answer in this exact structure (use the headings verbatim):

1) EXECUTIVE SUMMARY — 2-3 sentences, name the strongest and weakest period, mention per-day if lengths differ.
2) WHAT DROVE THE CHANGE — 2-4 bullets, each tied to a specific dimension/member with numbers and CV if relevant.
3) RISKS & WATCH-OUTS — 2-3 bullets, concentration or volatility concerns with numbers.
4) OPPORTUNITIES — 2-3 bullets, where to double-down, with the metric and period.
5) NEXT ACTIONS — 3 concrete steps a {role} should take this week, each starting with a verb. Include a check on data coverage if any period looks thin.

Rules:
- Quote only numbers that appear in the KPIS / DIMENSIONAL sections above.
- Never invent a figure, product, region or channel.
- Keep every bullet under 28 words.
- Use NPR formatting (e.g. NPR 1,240,000).
- Be specific and commercial, not generic.
- If per-day and total tell different stories (e.g. Feb shorter than Jan), call it out.
"""

    system_prompt = (
        "You are InsightFlow's comparison analyst. You write for business operators who need "
        "decisions, not dashboards. Every number you quote must come from the comparison data "
        "provided. If the comparison shows no material change, say so honestly. Favor per-day "
        "comparisons when period lengths differ."
    )

    try:
        from app.core.config import get_settings

        settings = get_settings()
        has_provider = bool(settings.groq_api_key or settings.gemini_api_key)
        if has_provider:
            from app.services.ai.provider import AIMessage, get_ai_response

            reply = await get_ai_response(
                [AIMessage(role="user", content=prompt)],
                system_prompt=system_prompt,
            )
            if reply and reply.strip():
                sections = _parse_ai_sections(reply)
                return {
                    "summary": sections.get("summary") or reply[:600],
                    "narrative": reply,
                    "sections": sections,
                    "source": "llm",
                    "disclaimer": "AI-generated — verify before acting.",
                }
    except Exception as e:
        logger.warning("compare AI suggestions failed, using deterministic fallback: %s", e)

    highlights = insights.get("highlights", [])
    drivers = insights.get("drivers", [])
    watchouts = insights.get("watchouts", [])

    fallback_sections = {
        "summary": insights.get("verdict", "Comparison complete."),
        "what_drove": drivers[:4] or ["No single dimension dominates the change — movement is spread across members."],
        "risks": watchouts[:3] or ["No major concentration or volatility flagged in the selected dimensions."],
        "opportunities": highlights[:3] or ["Review the dimensional tables for member-level opportunities."],
        "next_actions": [
            "Drill into the top driver in the dimensional table and check its underlying transactions.",
            "Validate the weakest period against data-coverage — confirm it is fully loaded, not truncated.",
            "Share this comparison with the relevant owner and set a follow-up KPI target for the next period.",
        ],
    }
    narrative = (
        f"**Executive Summary**\n{fallback_sections['summary']}\n\n"
        f"**What drove the change**\n" + "\n".join(f"- {b}" for b in fallback_sections["what_drove"]) + "\n\n"
        "**Risks & watch-outs**\n" + "\n".join(f"- {b}" for b in fallback_sections["risks"]) + "\n\n"
        "**Opportunities**\n" + "\n".join(f"- {b}" for b in fallback_sections["opportunities"]) + "\n\n"
        "**Next actions**\n" + "\n".join(f"- {b}" for b in fallback_sections["next_actions"])
    )
    return {
        "summary": fallback_sections["summary"],
        "narrative": narrative,
        "sections": fallback_sections,
        "source": "deterministic",
        "disclaimer": "Deterministic summary — no LLM was used. Add an API key for AI-enriched narrative.",
    }


def _parse_ai_sections(reply: str) -> dict[str, Any]:
    """Split a structured LLM reply into named sections (best-effort)."""
    import re

    sections: dict[str, Any] = {}
    patterns = {
        "summary": r"(?:EXECUTIVE SUMMARY|SUMMARY)[:\-]?\s*(.+?)(?=\n\s*\d+\)|\n\s*WHAT DROVE|\Z)",
        "what_drove": r"WHAT DROVE[^:\n]*[:\-]?\s*(.+?)(?=\n\s*\d+\)|\n\s*RISKS|\Z)",
        "risks": r"RISKS[^:\n]*[:\-]?\s*(.+?)(?=\n\s*\d+\)|\n\s*OPPORTUNITIES|\Z)",
        "opportunities": r"OPPORTUNITIES[:\-]?\s*(.+?)(?=\n\s*\d+\)|\n\s*NEXT ACTIONS|\Z)",
        "next_actions": r"NEXT ACTIONS[:\-]?\s*(.+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, reply, re.IGNORECASE | re.DOTALL)
        if m:
            raw = m.group(1).strip()
            bullets = [re.sub(r"^[\-\*\•\d\.\)\s]+", "", ln).strip() for ln in raw.splitlines() if ln.strip()]
            bullets = [b for b in bullets if b]
            if key == "summary":
                sections[key] = bullets[0] if bullets else raw[:400]
            else:
                sections[key] = bullets[:5] if bullets else [raw[:280]]
    if not sections:
        first_para = reply.strip().split("\n\n")[0][:500]
        sections["summary"] = first_para
        sections["narrative"] = reply
    else:
        sections["narrative"] = reply
    return sections
