"""Customer / product / region segmentation via K-Means + PCA (Power BI style).

Clusters entities on their behavioural features (revenue, volume, margin, AOV)
and projects them to 2-D with PCA so the UI can render a segment scatter.
Every cluster is summarised with its centroid profile so the analyst can name
it ("high-value low-volume", "steady mid-tier", ...).
"""

import logging
from typing import Any

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Product, SalesTransaction
from app.services.analytics.queries import Filters

logger = logging.getLogger(__name__)


async def _entity_frame(db: AsyncSession, f: Filters, dimension: str) -> tuple[list[str], np.ndarray]:
    """Return (labels, feature_matrix) for the chosen dimension."""
    from app.services.analytics.queries import _sales_conditions

    if dimension == "product":
        key, grp = Product.name, [Product.name]
        join = SalesTransaction.__table__.join(Product.__table__, Product.id == SalesTransaction.product_id)
    elif dimension == "category":
        key, grp = Product.category, [Product.category]
        join = SalesTransaction.__table__.join(Product.__table__, Product.id == SalesTransaction.product_id)
    elif dimension == "region":
        key, grp = SalesTransaction.region, [SalesTransaction.region]
        join = SalesTransaction.__table__
    elif dimension == "channel":
        key, grp = SalesTransaction.channel, [SalesTransaction.channel]
        join = SalesTransaction.__table__
    else:
        raise ValueError(f"unknown dimension {dimension}")

    stmt = (
        select(
            key.label("key"),
            func.sum(SalesTransaction.total_amount).label("revenue"),
            func.sum(SalesTransaction.quantity).label("units"),
            func.count(SalesTransaction.id).label("orders"),
            func.avg(SalesTransaction.unit_price).label("avg_price"),
            func.sum(
                SalesTransaction.total_amount - func.coalesce(Product.unit_cost, 0) * SalesTransaction.quantity
            ).label("margin"),
        )
        .select_from(join)
        .where(*_sales_conditions(f, f.date_from, f.date_to))
        .group_by(*grp)
    )
    rows = (await db.execute(stmt)).all()
    labels = [r.key or "(unknown)" for r in rows]
    feats = []
    for r in rows:
        revenue = float(r.revenue or 0.0)
        units = float(r.units or 0.0)
        orders = float(r.orders or 0.0)
        margin = float(r.margin or 0.0)
        aov = revenue / orders if orders else 0.0
        margin_pct = (margin / revenue * 100.0) if revenue else 0.0
        feats.append([revenue, units, margin_pct, aov])
    return labels, np.array(feats, dtype=float)


async def segment(
    db: AsyncSession,
    f: Filters,
    dimension: str = "product",
    n_clusters: int = 4,
) -> dict[str, Any]:
    """K-Means segmentation projected to 2-D via PCA."""
    labels, X = await _entity_frame(db, f, dimension)
    if len(labels) < max(3, n_clusters):
        return {
            "dimension": dimension,
            "n_clusters": 0,
            "entities": [],
            "clusters": [],
            "note": "not enough distinct entities to cluster",
        }

    # Standardise so no single large-magnitude feature dominates the distance.
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xs = (X - mean) / std

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    k = min(n_clusters, len(labels) - 1)
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(Xs)
    pca = PCA(n_components=2).fit(Xs)
    coords = pca.transform(Xs)

    entities = [
        {
            "label": labels[i],
            "cluster": int(km.labels_[i]),
            "x": round(float(coords[i, 0]), 3),
            "y": round(float(coords[i, 1]), 3),
            "revenue": round(float(X[i, 0]), 2),
            "units": round(float(X[i, 1]), 2),
            "margin_pct": round(float(X[i, 2]), 1),
            "aov": round(float(X[i, 3]), 2),
        }
        for i in range(len(labels))
    ]

    clusters = []
    for c in range(k):
        members = [e for e in entities if e["cluster"] == c]
        if not members:
            continue
        clusters.append(
            {
                "cluster": c,
                "size": len(members),
                "avg_revenue": round(float(np.mean([m["revenue"] for m in members])), 2),
                "avg_margin_pct": round(float(np.mean([m["margin_pct"] for m in members])), 1),
                "avg_units": round(float(np.mean([m["units"] for m in members])), 2),
                "avg_aov": round(float(np.mean([m["aov"] for m in members])), 2),
                "members_sample": [m["label"] for m in members[:5]],
            }
        )
    clusters.sort(key=lambda c: c["avg_revenue"], reverse=True)

    return {
        "dimension": dimension,
        "n_clusters": k,
        "pca_variance_explained": [round(float(v), 3) for v in pca.explained_variance_ratio_],
        "entities": entities,
        "clusters": clusters,
    }
