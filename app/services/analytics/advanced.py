"""Power BI-grade analytical shapes built on the warehouse.

Every function accepts the shared ``Filters`` object so the same global
cross-filters (date / region / channel / category) that drive the rest of the
dashboard also drive these visuals. All shapes are plain dicts the frontend
renders with bespoke SVG components.
"""

import logging
from typing import Any

import numpy as np
from sqlalchemy import Date, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Product, SalesTransaction
from app.services.analytics.cache import cached_query
from app.services.analytics.queries import Filters

logger = logging.getLogger(__name__)

DIMS = ("region", "channel", "category", "product")

_METRIC_AGG = {
    "revenue": func.sum(SalesTransaction.total_amount),
    "orders": func.count(SalesTransaction.id),
    "gross_margin": func.sum(
        SalesTransaction.total_amount - func.coalesce(Product.unit_cost, 0) * SalesTransaction.quantity
    ),
    "avg_order_value": func.avg(SalesTransaction.total_amount),
    "units": func.sum(SalesTransaction.quantity),
}

ALLOWED_METRICS = set(_METRIC_AGG.keys())
ALLOWED_GRANULARITIES = {"day", "week", "month", "quarter", "year"}

def _col(dim: str):
    if dim == "product":
        return Product.name
    if dim == "category":
        return Product.category
    return getattr(SalesTransaction, dim)


def _join_for(*dims):
    if "product" in dims or "category" in dims:
        return SalesTransaction.__table__.join(Product.__table__, Product.id == SalesTransaction.product_id)
    return SalesTransaction.__table__


def _conditions(f: Filters):
    from app.services.analytics.queries import _sales_conditions

    return _sales_conditions(f, f.date_from, f.date_to)


def _validate_metric(metric: str, fallback: str = "revenue") -> str:
    if metric in ALLOWED_METRICS:
        return metric
    return fallback


def _validate_granularity(g: str, fallback: str = "month") -> str:
    if g in ALLOWED_GRANULARITIES:
        return g
    return fallback


def _validate_dim(dim: str, fallback: str = "region") -> str:
    if dim in DIMS:
        return dim
    return fallback


# ── Decomposition tree ──────────────────────────────────────────────
@cached_query(ttl_seconds=30)
async def decomposition_tree(
    db: AsyncSession, f: Filters, metric: str = "revenue", hierarchy: str = "region,category,product"
) -> dict:
    metric = _validate_metric(metric)
    levels = [d.strip() for d in hierarchy.split(",") if d.strip() in DIMS]
    if not levels:
        levels = ["region", "category"]
    cols = [_col(d).label(f"d{i}") for i, d in enumerate(levels)]
    metric_expr = _METRIC_AGG[metric]
    stmt = (
        select(*cols, func.sum(metric_expr).label("value"))
        .select_from(_join_for(*levels))
        .where(*_conditions(f))
        .group_by(*[c.element for c in cols])
    )
    rows = (await db.execute(stmt)).all()
    tree: dict[str, Any] = {"name": "Total", "children": {}}
    total = 0.0
    for r in rows:
        vals = [getattr(r, f"d{i}") or "(unknown)" for i in range(len(levels))]
        v = float(r.value or 0.0)
        total += v
        node = tree
        for key in vals:
            node["children"].setdefault(key, {"name": key, "children": {}})
            node = node["children"][key]
        node["value"] = node.get("value", 0.0) + v

    def finalize(node: dict) -> dict:
        children = list(node.get("children", {}).values())
        value = node.get("value", sum(c.get("value", 0.0) for c in children))
        out = {
            "name": node["name"],
            "value": round(value, 2),
            "share_pct": round(value / total * 100, 1) if total else 0.0,
        }
        if children:
            out["children"] = sorted([finalize(c) for c in children], key=lambda c: c["value"], reverse=True)
        return out

    return {"metric": metric, "hierarchy": levels, "root": finalize(tree), "total": round(total, 2)}


# ── Waterfall / variance bridge ─────────────────────────────────────
@cached_query(ttl_seconds=30)
async def waterfall(
    db: AsyncSession, f: Filters, metric: str = "revenue", dimension: str = "category", top_n: int = 8
) -> dict:
    metric = _validate_metric(metric)
    dimension = _validate_dim(dimension, "category")
    cur_stmt = (
        select(_col(dimension).label("k"), func.sum(_METRIC_AGG[metric]).label("v"))
        .select_from(_join_for(dimension))
        .where(*_conditions(f))
        .group_by(_col(dimension))
    )
    prev_from, prev_to = f.previous_period()
    pf = Filters(
        date_from=prev_from,
        date_to=prev_to,
        region=f.region,
        channel=f.channel,
        category=f.category,
        regions=f.regions,
        channels=f.channels,
        categories=f.categories,
        org_id=f.org_id,
    )
    prev_stmt = (
        select(_col(dimension).label("k"), func.sum(_METRIC_AGG[metric]).label("v"))
        .select_from(_join_for(dimension))
        .where(*_conditions(pf))
        .group_by(_col(dimension))
    )
    cur = {r.k or "(unknown)": float(r.v or 0) for r in (await db.execute(cur_stmt)).all()}
    prev = {r.k or "(unknown)": float(r.v or 0) for r in (await db.execute(prev_stmt)).all()}
    keys = list({*cur, *prev})
    steps = []
    for k in keys:
        delta = cur.get(k, 0.0) - prev.get(k, 0.0)
        # change_pct per step relative to start
        start_total = sum(prev.values())
        change_pct = (delta / start_total * 100) if start_total else None
        steps.append({"label": k, "delta": round(delta, 2), "prev": round(prev.get(k, 0.0), 2), "cur": round(cur.get(k, 0.0), 2), "change_pct": round(change_pct, 1) if change_pct is not None else None})
    # split positive/negative, cumulative, then recombine
    positives = sorted([s for s in steps if s["delta"] >= 0], key=lambda s: s["delta"], reverse=True)
    negatives = sorted([s for s in steps if s["delta"] < 0], key=lambda s: s["delta"])  # most negative first
    # take top_n balanced: half positive, half negative, or proportional
    if top_n >= 2:
        half = top_n // 2
        # Ensure we cover both signs: take top half positives and top half negatives
        selected_pos = positives[:half + (top_n % 2)]
        selected_neg = negatives[:half]
        # If one side has fewer than half, fill from other side
        if len(selected_pos) < half:
            need = half - len(selected_pos)
            selected_neg = negatives[: half + need + (top_n % 2)]
        if len(selected_neg) < half:
            need = half - len(selected_neg)
            selected_pos = positives[: half + need + (top_n % 2)]
        selected = selected_pos + selected_neg
        # sort selected by delta descending for cumulative? But cumulative needs sequential order.
        # For waterfall, order by absolute delta magnitude descending, then compute cumulative
        selected_sorted = sorted(selected, key=lambda s: s["delta"], reverse=True)
        # recompute cumulative
        start = round(sum(prev.values()), 2)
        end = round(sum(cur.values()), 2)
        cum = start
        for s in selected_sorted:
            cum += s["delta"]
            s["cumulative"] = round(cum, 2)
            s["start"] = round(cum - s["delta"], 2)
            s["end"] = round(cum, 2)
        # final sort for display: largest positive first, then negatives at bottom? Keep delta desc
        selected_sorted.sort(key=lambda s: s["delta"], reverse=True)
        steps_out = selected_sorted[:top_n]
        # If top_n larger than selected, fill remaining by next largest absolute
        if len(steps_out) < top_n:
            remaining = [s for s in steps if s not in selected]
            remaining_sorted = sorted(remaining, key=lambda s: abs(s["delta"]), reverse=True)
            extra = remaining_sorted[: top_n - len(steps_out)]
            for s in extra:
                cum += s["delta"]
                s["cumulative"] = round(cum, 2)
                s["start"] = round(cum - s["delta"], 2)
                s["end"] = round(cum, 2)
            steps_out.extend(extra)
            steps_out.sort(key=lambda s: s["delta"], reverse=True)
    else:
        steps.sort(key=lambda s: s["delta"], reverse=True)
        steps_out = steps[:top_n]
        cum = sum(prev.values())
        for s in steps_out:
            start_cum = cum
            cum += s["delta"]
            s["cumulative"] = round(cum, 2)
            s["start"] = round(start_cum, 2)
            s["end"] = round(cum, 2)
    start = round(sum(prev.values()), 2)
    end = round(sum(cur.values()), 2)
    # Ensure steps_out cumulative is consistent (recompute if needed)
    if steps_out and "cumulative" not in steps_out[0]:
        cum = start
        for s in steps_out:
            cum += s["delta"]
            s["cumulative"] = round(cum, 2)
    return {
        "metric": metric,
        "dimension": dimension,
        "start": start,
        "end": end,
        "total_change": round(end - start, 2),
        "change_pct": round((end - start) / start * 100, 1) if start else None,
        "steps": steps_out,
    }


# ── Heatmap matrix ─────────────────────────────────────────────────
@cached_query(ttl_seconds=30)
async def heatmap(
    db: AsyncSession, f: Filters, metric: str = "revenue", row_dim: str = "region", col_dim: str = "category"
) -> dict:
    metric = _validate_metric(metric)
    row_dim = _validate_dim(row_dim, "region")
    col_dim = _validate_dim(col_dim, "category")
    if row_dim not in DIMS or col_dim not in DIMS:
        row_dim, col_dim = "region", "category"
    rcol, ccol = _col(row_dim), _col(col_dim)
    stmt = (
        select(rcol.label("r"), ccol.label("c"), func.sum(_METRIC_AGG[metric]).label("v"))
        .select_from(_join_for(row_dim, col_dim))
        .where(*_conditions(f))
        .group_by(rcol, ccol)
    )
    rows = (await db.execute(stmt)).all()
    data: dict[tuple, float] = {(r.r or "(unknown)", r.c or "(unknown)"): float(r.v or 0) for r in rows}
    r_keys = sorted({k[0] for k in data})
    c_keys = sorted({k[1] for k in data})
    matrix = [[round(data.get((rk, ck), 0.0), 2) for ck in c_keys] for rk in r_keys]
    flat = [v for row in matrix for v in row] or [0]
    # p95 clipping for color scale — prevents one outlier washing out the heatmap
    if flat:
        arr = np.array(flat, dtype=float)
        p95 = float(np.percentile(arr, 95)) if len(arr) > 1 else float(arr[0])
        p5 = float(np.percentile(arr, 5)) if len(arr) > 1 else 0.0
        vmax = p95 if p95 > 0 else max(flat)
        vmin = p5
        # clipped matrix for display scaling (original values kept in matrix)
        clipped = np.clip(arr, vmin, vmax)
        display_max = float(vmax)
        display_min = float(vmin)
    else:
        display_max = 0.0
        display_min = 0.0
        clipped = np.array([])
    return {
        "metric": metric,
        "row_dim": row_dim,
        "col_dim": col_dim,
        "rows": r_keys,
        "cols": c_keys,
        "matrix": matrix,
        "min": min(flat) if flat else 0,
        "max": max(flat) if flat else 0,
        "p95": round(float(p95), 2) if flat and len(flat) > 1 else (max(flat) if flat else 0),
        "p5": round(float(p5), 2) if flat and len(flat) > 1 else 0,
        "display_min": round(display_min, 2),
        "display_max": round(display_max, 2),
    }


# ── Scatter / bubble ───────────────────────────────────────────────
@cached_query(ttl_seconds=30)
async def scatter(
    db: AsyncSession,
    f: Filters,
    dimension: str = "product",
    x: str = "revenue",
    y: str = "margin_pct",
    size: str = "units",
) -> dict:
    dimension = _validate_dim(dimension, "product")
    # whitelist metrics for axes
    allowed_axes = {"revenue", "units", "orders", "aov", "gross_margin", "margin_pct", "margin", "avg_order_value"}
    # normalize axis names
    # map frontend names to internal point keys
    # points have keys: revenue, units, orders, aov, gross_margin, margin_pct
    # accept both
    if x not in allowed_axes:
        x = "revenue"
    if y not in allowed_axes:
        y = "margin_pct"
    if size not in allowed_axes:
        size = "units"
    key = _col(dimension)
    stmt = (
        select(
            key.label("label"),
            func.sum(SalesTransaction.total_amount).label("revenue"),
            func.sum(SalesTransaction.quantity).label("units"),
            func.count(SalesTransaction.id).label("orders"),
            func.avg(SalesTransaction.total_amount).label("aov"),
            func.sum(
                SalesTransaction.total_amount - func.coalesce(Product.unit_cost, 0) * SalesTransaction.quantity
            ).label("gross_margin"),
        )
        .select_from(_join_for(dimension))
        .where(*_conditions(f))
        .group_by(key)
    )
    # Fix GROUP BY when adding category: need to group by both key and category
    cat_col = None
    if dimension in ("category", "product"):
        cat_col = Product.category.label("cat")
        stmt = stmt.add_columns(cat_col)
        # Need to group by both key and product category to satisfy SQL
        # For product dimension, category is functionally dependent but SQL requires it
        try:
            stmt = stmt.group_by(key, Product.category)
        except Exception:
            # if already grouped, reconstruct
            stmt = stmt.group_by(key, Product.category)
    else:
        # ensure group_by only key
        pass
    rows = (await db.execute(stmt)).all()
    pts = []
    for r in rows:
        revenue = float(r.revenue or 0)
        units = float(r.units or 0)
        orders = float(r.orders or 0)
        margin = float(r.gross_margin or 0)
        margin_pct = (margin / revenue * 100) if revenue else 0.0
        aov = revenue / orders if orders else 0.0
        pts.append(
            {
                "label": r.label or "(unknown)",
                "category": getattr(r, "cat", None),
                "revenue": round(revenue, 2),
                "units": round(units, 2),
                "orders": int(orders),
                "aov": round(aov, 2),
                "gross_margin": round(margin, 2),
                "margin_pct": round(margin_pct, 1),
                "margin": round(margin, 2),
                "avg_order_value": round(aov, 2),
            }
        )
    picks = {"x": x, "y": y, "size": size}

    def pick(p, field):
        # map aliases
        if field == "margin":
            field = "gross_margin"
        v = p.get(field)
        return float(v if v is not None else 0.0)

    # handle empty pts
    if pts:
        x_vals = [pick(p, x) for p in pts]
        y_vals = [pick(p, y) for p in pts]
        x_range = [round(float(min(x_vals)), 2), round(float(max(x_vals)), 2)]
        y_range = [round(float(min(y_vals)), 2), round(float(max(y_vals)), 2)]
    else:
        x_range = [0, 0]
        y_range = [0, 0]
    return {
        "dimension": dimension,
        "axes": picks,
        "points": pts,
        "x_range": x_range,
        "y_range": y_range,
    }


# ── Funnel ─────────────────────────────────────────────────────────
@cached_query(ttl_seconds=30)
async def funnel(
    db: AsyncSession, f: Filters, metric: str = "revenue", dimension: str = "category", top_n: int = 8
) -> dict:
    metric = _validate_metric(metric)
    dimension = _validate_dim(dimension, "category")
    stmt = (
        select(_col(dimension).label("k"), func.sum(_METRIC_AGG[metric]).label("v"))
        .select_from(_join_for(dimension))
        .where(*_conditions(f))
        .group_by(_col(dimension))
        .order_by(func.sum(_METRIC_AGG[metric]).desc())
    )
    rows = (await db.execute(stmt)).all()
    stages = [{"label": r.k or "(unknown)", "value": round(float(r.v or 0), 2)} for r in rows][:top_n]
    return {"metric": metric, "dimension": dimension, "stages": stages}


# ── Radar (multi-metric comparison of entities) ────────────────────
@cached_query(ttl_seconds=30)
async def radar(
    db: AsyncSession, f: Filters, dimension: str = "region", metrics: str = "revenue,orders,gross_margin,aov,units"
) -> dict:
    dimension = _validate_dim(dimension, "region")
    metric_list = [m.strip() for m in metrics.split(",") if m.strip() in _METRIC_AGG]
    if not metric_list:
        metric_list = ["revenue", "orders"]
    key = _col(dimension)
    cols = [key.label("k")] + [func.sum(_METRIC_AGG[m]).label(m) for m in metric_list]
    stmt = select(*cols).select_from(_join_for(dimension)).where(*_conditions(f)).group_by(key)
    rows = (await db.execute(stmt)).all()
    entities = [r.k or "(unknown)" for r in rows]
    raw = {m: np.array([float(getattr(r, m) or 0) for r in rows]) for m in metric_list}
    series = {}
    for m, arr in raw.items():
        lo, hi = (arr.min(), arr.max()) if len(arr) else (0, 0)
        rng = (hi - lo) or 1.0
        series[m] = ((arr - lo) / rng * 100).tolist()
    out = []
    for i, e in enumerate(entities):
        out.append(
            {
                "entity": e,
                "normalized": {m: round(float(series[m][i]), 1) for m in metric_list},
                "raw": {m: round(float(raw[m][i]), 2) for m in metric_list},
            }
        )
    return {"dimension": dimension, "axes": metric_list, "entities": out}


# ── Small multiples (metric trend split by dimension) ──────────────
@cached_query(ttl_seconds=30)
async def small_multiples(
    db: AsyncSession, f: Filters, metric: str = "revenue", dimension: str = "region", granularity: str = "month"
) -> dict:
    metric = _validate_metric(metric)
    dimension = _validate_dim(dimension, "region")
    granularity = _validate_granularity(granularity, "month")
    key = _col(dimension)
    bucket = cast(func.date_trunc(granularity, cast(SalesTransaction.txn_date, Date)), Date)
    stmt = (
        select(bucket.label("period"), key.label("k"), func.sum(_METRIC_AGG[metric]).label("v"))
        .select_from(_join_for(dimension))
        .where(*_conditions(f))
        .group_by(bucket, key)
        .order_by(bucket, key)
    )
    rows = (await db.execute(stmt)).all()
    members: dict[str, list[dict]] = {}
    periods_set: set = set()
    for r in rows:
        k = r.k or "(unknown)"
        # store raw
        members.setdefault(k, []).append({"period": str(r.period), "value": round(float(r.v or 0), 2), "_date": r.period})
        periods_set.add(r.period)
    # zero-fill: ensure every member has an entry for every period
    sorted_periods = sorted(periods_set)
    sorted_period_str = [str(p) for p in sorted_periods]
    # Build lookup per member
    filled_members: dict[str, list[dict]] = {}
    for member, pts in members.items():
        lookup = {p["_date"]: p["value"] for p in pts}
        filled = []
        for p in sorted_periods:
            filled.append({"period": str(p), "value": round(float(lookup.get(p, 0.0)), 2)})
        filled_members[member] = filled
    # also handle case where no rows: still return periods from f date range
    if not sorted_periods:
        # try to generate periods from filter range
        try:
            from datetime import timedelta as td
            # generate based on granularity
            # simple fallback: empty
            pass
        except Exception:
            pass
    return {
        "metric": metric,
        "dimension": dimension,
        "granularity": granularity,
        "periods": sorted_period_str,
        "series": [
            {"member": m, "points": pts}
            for m, pts in sorted(filled_members.items(), key=lambda kv: -sum(p["value"] for p in kv[1]))
        ],
    }
