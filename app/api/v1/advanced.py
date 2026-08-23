"""Power BI-grade analytics + advanced prediction endpoints.

Reuses the shared ``get_filters`` dependency so every visual responds to the
global cross-filters (date / region / channel / category) exactly like the
rest of the dashboard.
"""

from datetime import date

from fastapi import APIRouter, Depends, Query
from sqlalchemy import Date, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import DbSession, get_current_user
from app.api.v1.analytics import FiltersDep, get_filters
from app.core.clock import business_today
from app.models import Product, SalesTransaction
from app.services.analytics import advanced
from app.services.analytics.influencers import key_influencers
from app.services.analytics.queries import Filters
from app.services.ml import forecasting as fc
from app.services.ml.scenario import monte_carlo
from app.services.ml.segmentation import segment

router = APIRouter(prefix="/advanced", tags=["advanced"], dependencies=[Depends(get_current_user)])


def _daily_series(db: AsyncSession, f: Filters, metric: str = "revenue") -> list[dict]:
    bucket = cast(func.date_trunc("day", cast(SalesTransaction.txn_date, Date)), Date)
    if metric == "orders":
        expr = func.count(SalesTransaction.id)
    elif metric == "gross_margin":
        expr = func.sum(
            SalesTransaction.total_amount
            - func.coalesce(Product.unit_cost, 0) * SalesTransaction.quantity
        )
    else:
        expr = func.sum(SalesTransaction.total_amount)
    from app.services.analytics.queries import _sales_conditions

    stmt = (
        select(bucket.label("ds"), expr.label("y"))
        .select_from(
            SalesTransaction.__table__.join(
                Product.__table__, Product.id == SalesTransaction.product_id, isouter=True
            )
        )
        .where(*_sales_conditions(f, f.date_from, f.date_to))
        .group_by(bucket)
        .order_by(bucket)
    )
    rows = db.execute(stmt).all()
    return [{"ds": r.ds, "y": float(r.y or 0.0)} for r in rows]


@router.get("/decomposition-tree")
async def decomposition_tree_endpoint(
    db: DbSession, f: FiltersDep, metric: str = "revenue", hierarchy: str = "region,category,product"
):
    return await advanced.decomposition_tree(db, f, metric=metric, hierarchy=hierarchy)


@router.get("/waterfall")
async def waterfall_endpoint(
    db: DbSession, f: FiltersDep, metric: str = "revenue", dimension: str = "category", top_n: int = 8
):
    return await advanced.waterfall(db, f, metric=metric, dimension=dimension, top_n=top_n)


@router.get("/heatmap")
async def heatmap_endpoint(
    db: DbSession, f: FiltersDep, metric: str = "revenue", row_dim: str = "region", col_dim: str = "category"
):
    return await advanced.heatmap(db, f, metric=metric, row_dim=row_dim, col_dim=col_dim)


@router.get("/scatter")
async def scatter_endpoint(
    db: DbSession,
    f: FiltersDep,
    dimension: str = "product",
    x: str = "revenue",
    y: str = "margin_pct",
    size: str = "units",
):
    return await advanced.scatter(db, f, dimension=dimension, x=x, y=y, size=size)


@router.get("/funnel")
async def funnel_endpoint(
    db: DbSession, f: FiltersDep, metric: str = "revenue", dimension: str = "category", top_n: int = 8
):
    return await advanced.funnel(db, f, metric=metric, dimension=dimension, top_n=top_n)


@router.get("/radar")
async def radar_endpoint(
    db: DbSession, f: FiltersDep, dimension: str = "region", metrics: str = "revenue,orders,gross_margin,aov,units"
):
    return await advanced.radar(db, f, dimension=dimension, metrics=metrics)


@router.get("/small-multiples")
async def small_multiples_endpoint(
    db: DbSession,
    f: FiltersDep,
    metric: str = "revenue",
    dimension: str = "region",
    granularity: str = "month",
):
    return await advanced.small_multiples(db, f, metric=metric, dimension=dimension, granularity=granularity)


@router.get("/key-influencers")
async def key_influencers_endpoint(db: DbSession, f: FiltersDep, target: str = "revenue"):
    return await key_influencers(db, f, target=target)


@router.get("/segmentation")
async def segmentation_endpoint(db: DbSession, f: FiltersDep, dimension: str = "product", n_clusters: int = 4):
    return await segment(db, f, dimension=dimension, n_clusters=n_clusters)


@router.get("/forecast-scenarios")
async def forecast_scenarios_endpoint(
    db: DbSession,
    f: FiltersDep,
    metric: str = "revenue",
    horizon: int = Query(30, ge=1, le=180),
    n_paths: int = Query(500, ge=50, le=2000),
    model: str | None = None,
):
    series = _daily_series(db, f, metric=metric)
    if len(series) < 7:
        return {"error": "insufficient history for probabilistic forecast", "points": 0}
    import pandas as pd

    frame = pd.DataFrame(series)
    return monte_carlo(frame, horizon=horizon, n_paths=n_paths, model=model)


@router.get("/model-comparison")
async def model_comparison_endpoint(
    db: DbSession, f: FiltersDep, metric: str = "revenue"
):
    """Holdout MAPE for every candidate model on the selected series."""
    series = _daily_series(db, f, metric=metric)
    if len(series) < fc.HOLDOUT_DAYS + 14:
        return {"error": "need at least ~104 days of history", "candidates": []}
    import pandas as pd

    frame = pd.DataFrame(series)
    evals = fc.evaluate_candidates(frame)
    best_name, _ = fc.best_candidate(frame)
    return {
        "metric": metric,
        "candidates": [
            {"model": e.model_name, "mape": e.metrics["mape"], "rmse": e.metrics["rmse"], "mae": e.metrics["mae"]}
            for e in evals
        ],
        "best": best_name,
    }
