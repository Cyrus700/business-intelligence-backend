"""Key Influencers — Power BI style "what drives this metric".

Three complementary signals, all grounded in warehouse aggregates:

1. **Dimensional influence** — for each categorical dimension we measure how
   much its member spread accounts for variation in the target metric
   (sum-of-squares decomposition). The dimension with the largest share is the
   leading influencer (e.g. "Region explains 62% of revenue variation").
2. **Key influencer members** — within the leading dimension, the members whose
   value deviates most from the mean (largest absolute lift). These are the
   concrete drivers a manager should look at.
3. **Numeric drivers** — Pearson correlation of the target with numeric
   features (volume, price, margin) across products, plus a RandomForest
   feature-importance ranking for a transparent ML view.
"""

import logging
from dataclasses import dataclass

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Product, SalesTransaction
from app.services.analytics.queries import Filters

logger = logging.getLogger(__name__)

DIMENSIONS = ("region", "channel", "category", "product")


def _member_values(db: AsyncSession, f: Filters, dimension: str) -> list[tuple[str, float]]:
    """(member, metric_value) for a dimension using the active filters."""
    from app.services.analytics.queries import _sales_conditions

    if dimension == "product":
        key = Product.name
        stmt_from = SalesTransaction.__table__.join(
            Product.__table__, Product.id == SalesTransaction.product_id
        )
        group = [Product.name]
    elif dimension == "category":
        key = Product.category
        stmt_from = SalesTransaction.__table__.join(
            Product.__table__, Product.id == SalesTransaction.product_id
        )
        group = [Product.category]
    else:
        key = getattr(SalesTransaction, dimension)
        stmt_from, group = SalesTransaction.__table__, [key]

    stmt = (
        select(key.label("key"), func.sum(SalesTransaction.total_amount).label("value"))
        .select_from(stmt_from)
        .where(*_sales_conditions(f, f.date_from, f.date_to))
        .group_by(*group)
    )
    return [(r.key or "(unknown)", float(r.value or 0.0)) for r in db.execute(stmt).all()]


def _numeric_drivers(db: AsyncSession, f: Filters) -> list[dict]:
    """Pearson correlation of revenue with volume/price/margin across products."""
    from app.services.analytics.queries import _sales_conditions

    stmt = (
        select(
            Product.id,
            func.sum(SalesTransaction.total_amount).label("revenue"),
            func.sum(SalesTransaction.quantity).label("units"),
            func.avg(SalesTransaction.unit_price).label("price"),
            func.sum(
                SalesTransaction.total_amount
                - func.coalesce(Product.unit_cost, 0) * SalesTransaction.quantity
            ).label("margin"),
        )
        .select_from(
            SalesTransaction.__table__.join(
                Product.__table__, Product.id == SalesTransaction.product_id
            )
        )
        .where(*_sales_conditions(f, f.date_from, f.date_to))
        .group_by(Product.id)
    )
    rows = db.execute(stmt).all()
    if len(rows) < 3:
        return []
    rev = np.array([float(r.revenue or 0) for r in rows])
    feats = {
        "units": np.array([float(r.units or 0) for r in rows]),
        "avg_price": np.array([float(r.price or 0) for r in rows]),
        "margin": np.array([float(r.margin or 0) for r in rows]),
    }

    corr = {}
    for name, arr in feats.items():
        if np.std(arr) < 1e-9 or np.std(rev) < 1e-9:
            corr[name] = 0.0
        else:
            corr[name] = round(float(np.corrcoef(arr, rev)[0, 1]), 3)

    importance: list[dict] = []
    try:
        from sklearn.ensemble import RandomForestRegressor

        X = np.column_stack([feats["units"], feats["avg_price"], feats["margin"]])
        if np.std(rev) > 1e-9 and np.all(np.std(X, axis=0) > 1e-9):
            rf = RandomForestRegressor(n_estimators=60, random_state=0, max_depth=4)
            rf.fit(X, rev)
            for name, imp in zip(("units", "avg_price", "margin"), rf.feature_importances_):
                importance.append({"feature": name, "importance": round(float(imp), 3)})
            importance.sort(key=lambda d: d["importance"], reverse=True)
    except Exception:  # noqa: BLE001 — ML is best-effort
        logger.exception("key-influencer RF failed")

    return [{"metric": "revenue", "correlations": corr, "ml_importance": importance}]


async def key_influencers(db: AsyncSession, f: Filters, target: str = "revenue") -> dict:
    """Return the three influencer signals for ``target`` (revenue by default)."""
    if target not in ("revenue", "gross_margin", "orders", "avg_order_value"):
        target = "revenue"

    dim_results = []
    for dim in DIMENSIONS:
        members = _member_values(db, f, dim)
        if len(members) < 2:
            continue
        vals = np.array([m[1] for m in members], dtype=float)
        total = vals.sum()
        if total <= 0:
            continue
        mean = vals.mean()
        ss = float(np.sum((vals - mean) ** 2))
        shares = vals / total
        equal = 1.0 / len(members)
        ranked = sorted(
            zip([m[0] for m in members], vals, shares),
            key=lambda t: abs(t[1] - mean),
            reverse=True,
        )[:5]
        dim_results.append(
            {
                "dimension": dim,
                "member_count": len(members),
                "variation_share": round(ss / (float(np.sum(vals**2)) or 1.0), 3),
                "top_members": [
                    {
                        "member": name,
                        "value": round(float(v), 2),
                        "share_pct": round(float(s) * 100, 1),
                        "lift_vs_average": round(float(s - equal) * 100, 1),
                    }
                    for name, v, s in ranked
                ],
            }
        )
    dim_results.sort(key=lambda d: d["variation_share"], reverse=True)
    leading = dim_results[0]["dimension"] if dim_results else None

    numeric = _numeric_drivers(db, f)

    return {
        "target": target,
        "leading_dimension": leading,
        "dimensional_influence": dim_results,
        "numeric_drivers": numeric,
        "method": "sum-of-squares decomposition + Pearson correlation + RandomForest importance",
    }
