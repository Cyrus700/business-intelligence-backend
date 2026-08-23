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
from app.services.analytics.queries import Filters

logger = logging.getLogger(__name__)

DIMS = ("region", "channel", "category", "product")

_METRIC_AGG = {
    "revenue": func.sum(SalesTransaction.total_amount),
    "orders": func.count(SalesTransaction.id),
    "gross_margin": func.sum(
        SalesTransaction.total_amount
        - func.coalesce(Product.unit_cost, 0) * SalesTransaction.quantity
    ),
    "avg_order_value": func.avg(SalesTransaction.total_amount),
    "units": func.sum(SalesTransaction.quantity),
}


def _col(dim: str):
    if dim == "product":
        return Product.name
    if dim == "category":
        return Product.category
    return getattr(SalesTransaction, dim)


def _join_for(*dims):
    if "product" in dims or "category" in dims:
        return SalesTransaction.__table__.join(
            Product.__table__, Product.id == SalesTransaction.product_id
        )
    return SalesTransaction.__table__


def _conditions(f: Filters):
    from app.services.analytics.queries import _sales_conditions

    return _sales_conditions(f, f.date_from, f.date_to)


# ── Decomposition tree ──────────────────────────────────────────────
async def decomposition_tree(
    db: AsyncSession, f: Filters, metric: str = "revenue", hierarchy: str = "region,category,product"
) -> dict:
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
    rows = db.execute(stmt).all()
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
            out["children"] = sorted(
                [finalize(c) for c in children], key=lambda c: c["value"], reverse=True
            )
        return out

    return {"metric": metric, "hierarchy": levels, "root": finalize(tree), "total": round(total, 2)}


# ── Waterfall / variance bridge ─────────────────────────────────────
async def waterfall(
    db: AsyncSession, f: Filters, metric: str = "revenue", dimension: str = "category", top_n: int = 8
) -> dict:
    cur_stmt = (
        select(_col(dimension).label("k"), func.sum(_METRIC_AGG[metric]).label("v"))
        .select_from(_join_for(dimension))
        .where(*_conditions(f))
        .group_by(_col(dimension))
    )
    prev_from, prev_to = f.previous_period()
    pf = Filters(date_from=prev_from, date_to=prev_to, region=f.region, channel=f.channel, category=f.category)
    prev_stmt = (
        select(_col(dimension).label("k"), func.sum(_METRIC_AGG[metric]).label("v"))
        .select_from(_join_for(dimension))
        .where(*_conditions(pf))
        .group_by(_col(dimension))
    )
    cur = {r.k or "(unknown)": float(r.v or 0) for r in db.execute(cur_stmt).all()}
    prev = {r.k or "(unknown)": float(r.v or 0) for r in db.execute(prev_stmt).all()}
    keys = list({*cur, *prev})
    steps = []
    for k in keys:
        delta = cur.get(k, 0.0) - prev.get(k, 0.0)
        steps.append({"label": k, "delta": round(delta, 2)})
    steps.sort(key=lambda s: s["delta"])
    start = round(sum(prev.values()), 2)
    end = round(sum(cur.values()), 2)
    return {
        "metric": metric,
        "dimension": dimension,
        "start": start,
        "end": end,
        "total_change": round(end - start, 2),
        "change_pct": round((end - start) / start * 100, 1) if start else None,
        "steps": steps[:top_n],
    }


# ── Heatmap matrix ─────────────────────────────────────────────────
async def heatmap(
    db: AsyncSession, f: Filters, metric: str = "revenue", row_dim: str = "region", col_dim: str = "category"
) -> dict:
    if row_dim not in DIMS or col_dim not in DIMS:
        row_dim, col_dim = "region", "category"
    rcol, ccol = _col(row_dim), _col(col_dim)
    stmt = (
        select(rcol.label("r"), ccol.label("c"), func.sum(_METRIC_AGG[metric]).label("v"))
        .select_from(_join_for(row_dim, col_dim))
        .where(*_conditions(f))
        .group_by(rcol, ccol)
    )
    rows = db.execute(stmt).all()
    data: dict[tuple, float] = {(r.r or "(unknown)", r.c or "(unknown)"): float(r.v or 0) for r in rows}
    r_keys = sorted({k[0] for k in data})
    c_keys = sorted({k[1] for k in data})
    matrix = [[round(data.get((rk, ck), 0.0), 2) for ck in c_keys] for rk in r_keys]
    flat = [v for row in matrix for v in row] or [0]
    return {
        "metric": metric,
        "row_dim": row_dim,
        "col_dim": col_dim,
        "rows": r_keys,
        "cols": c_keys,
        "matrix": matrix,
        "min": min(flat),
        "max": max(flat),
    }


# ── Scatter / bubble ───────────────────────────────────────────────
async def scatter(
    db: AsyncSession, f: Filters, dimension: str = "product", x: str = "revenue", y: str = "margin_pct", size: str = "units"
) -> dict:
    if dimension not in DIMS:
        dimension = "product"
    key = _col(dimension)
    stmt = (
        select(
            key.label("label"),
            func.sum(SalesTransaction.total_amount).label("revenue"),
            func.sum(SalesTransaction.quantity).label("units"),
            func.count(SalesTransaction.id).label("orders"),
            func.avg(SalesTransaction.total_amount).label("aov"),
            func.sum(
                SalesTransaction.total_amount
                - func.coalesce(Product.unit_cost, 0) * SalesTransaction.quantity
            ).label("gross_margin"),
        )
        .select_from(_join_for(dimension))
        .where(*_conditions(f))
        .group_by(key)
    )
    if dimension in ("category", "product"):
        stmt = stmt.add_columns(Product.category.label("cat"))
    rows = db.execute(stmt).all()
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
            }
        )
    picks = {"x": x, "y": y, "size": size}

    def pick(p, field):
        v = p.get(field)
        return float(v if v is not None else 0.0)

    return {
        "dimension": dimension,
        "axes": picks,
        "points": pts,
        "x_range": [min((pick(p, x) for p in pts), default=0), max((pick(p, x) for p in pts), default=0)],
        "y_range": [min((pick(p, y) for p in pts), default=0), max((pick(p, y) for p in pts), default=0)],
    }


# ── Funnel ─────────────────────────────────────────────────────────
async def funnel(db: AsyncSession, f: Filters, metric: str = "revenue", dimension: str = "category", top_n: int = 8) -> dict:
    stmt = (
        select(_col(dimension).label("k"), func.sum(_METRIC_AGG[metric]).label("v"))
        .select_from(_join_for(dimension))
        .where(*_conditions(f))
        .group_by(_col(dimension))
        .order_by(func.sum(_METRIC_AGG[metric]).desc())
    )
    rows = db.execute(stmt).all()
    stages = [{"label": r.k or "(unknown)", "value": round(float(r.v or 0), 2)} for r in rows][:top_n]
    return {"metric": metric, "dimension": dimension, "stages": stages}


# ── Radar (multi-metric comparison of entities) ────────────────────
async def radar(db: AsyncSession, f: Filters, dimension: str = "region", metrics: str = "revenue,orders,gross_margin,aov,units") -> dict:
    if dimension not in DIMS:
        dimension = "region"
    metric_list = [m.strip() for m in metrics.split(",") if m.strip() in _METRIC_AGG]
    if not metric_list:
        metric_list = ["revenue", "orders"]
    key = _col(dimension)
    cols = [key.label("k")] + [func.sum(_METRIC_AGG[m]).label(m) for m in metric_list]
    stmt = (
        select(*cols)
        .select_from(_join_for(dimension))
        .where(*_conditions(f))
        .group_by(key)
    )
    rows = db.execute(stmt).all()
    entities = [r.k or "(unknown)" for r in rows]
    raw = {m: np.array([float(getattr(r, m) or 0) for r in rows]) for m in metric_list}
    series = {}
    for m, arr in raw.items():
        lo, hi = (arr.min(), arr.max())
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
async def small_multiples(
    db: AsyncSession, f: Filters, metric: str = "revenue", dimension: str = "region", granularity: str = "month"
) -> dict:
    if dimension not in DIMS:
        dimension = "region"
    key = _col(dimension)
    bucket = cast(func.date_trunc(granularity, cast(SalesTransaction.txn_date, Date)), Date)
    stmt = (
        select(bucket.label("period"), key.label("k"), func.sum(_METRIC_AGG[metric]).label("v"))
        .select_from(_join_for(dimension))
        .where(*_conditions(f))
        .group_by(bucket, key)
        .order_by(bucket, key)
    )
    rows = db.execute(stmt).all()
    members: dict[str, list[dict]] = {}
    periods: set = set()
    for r in rows:
        k = r.k or "(unknown)"
        members.setdefault(k, []).append({"period": str(r.period), "value": round(float(r.v or 0), 2)})
        periods.add(str(r.period))
    return {
        "metric": metric,
        "dimension": dimension,
        "granularity": granularity,
        "periods": sorted(periods),
        "series": [
            {"member": m, "points": pts}
            for m, pts in sorted(members.items(), key=lambda kv: -sum(p["value"] for p in kv[1]))
        ],
    }
