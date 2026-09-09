"""Power BI-grade analytics + advanced prediction endpoints.

Reuses the shared ``get_filters`` dependency so every visual responds to the
global cross-filters (date / region / channel / category) exactly like the
rest of the dashboard.
"""

from datetime import date as _date

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import Date, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser, DbSession, get_current_user, is_super_admin
from app.api.v1.analytics import FiltersDep
from app.models import Product, SalesTransaction
from app.services.analytics import advanced
from app.services.analytics.cache import cached_query
from app.services.analytics.influencers import key_influencers
from app.services.analytics.queries import Filters
from app.services.ml import forecasting as fc
from app.services.ml.scenario import monte_carlo
from app.services.ml.segmentation import segment

router = APIRouter(prefix="/advanced", tags=["advanced"], dependencies=[Depends(get_current_user)])

ALLOWED_METRICS_SCENARIO = {"revenue", "orders", "gross_margin", "avg_order_value", "units"}
ALLOWED_GRANULARITY = {"day", "week", "month", "quarter", "year"}


def _scoped_filters(f: Filters, user: CurrentUser) -> Filters:
    import dataclasses

    if is_super_admin(user):
        return f
    return dataclasses.replace(f, org_id=user.org_id)


async def _daily_series(db: AsyncSession, f: Filters, metric: str = "revenue") -> list[dict]:
    # validate metric whitelist
    if metric not in ALLOWED_METRICS_SCENARIO and metric not in ("revenue", "orders", "gross_margin"):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"invalid metric '{metric}'")
    bucket = cast(func.date_trunc("day", cast(SalesTransaction.txn_date, Date)), Date)
    if metric == "orders":
        expr = func.count(SalesTransaction.id)
    elif metric == "gross_margin":
        expr = func.sum(SalesTransaction.total_amount - func.coalesce(Product.unit_cost, 0) * SalesTransaction.quantity)
    else:
        # revenue unified; avg_order_value handled via revenue/orders ratio? But daily series for forecasting expects y value
        # For avg_order_value we compute revenue per day / orders? Simplify to revenue
        if metric == "avg_order_value":
            expr = func.avg(SalesTransaction.total_amount)
        else:
            expr = func.sum(SalesTransaction.total_amount)
    from app.services.analytics.queries import _sales_conditions

    stmt = (
        select(bucket.label("ds"), expr.label("y"))
        .select_from(
            SalesTransaction.__table__.join(Product.__table__, Product.id == SalesTransaction.product_id, isouter=True)
        )
        .where(*_sales_conditions(f, f.date_from, f.date_to))
        .group_by(bucket)
        .order_by(bucket)
    )
    rows = (await db.execute(stmt)).all()
    raw = [{"ds": r.ds, "y": float(r.y or 0.0)} for r in rows]
    if not raw:
        return []
    # zero-fill with date_range
    from datetime import timedelta

    by_date = {r["ds"]: r["y"] for r in raw}
    # Use filter range for zero-fill, not just observed dates
    start = f.date_from
    end = f.date_to
    filled = []
    cur = start
    while cur <= end:
        filled.append({"ds": cur, "y": float(by_date.get(cur, 0.0))})
        cur += timedelta(days=1)
    return filled


@router.get("/decomposition-tree")
async def decomposition_tree_endpoint(
    db: DbSession, f: FiltersDep, user: CurrentUser, metric: str = "revenue", hierarchy: str = "region,category,product"
):
    f = _scoped_filters(f, user)
    if metric not in ALLOWED_METRICS_SCENARIO:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"invalid metric '{metric}'")
    return await advanced.decomposition_tree(db, f, metric=metric, hierarchy=hierarchy)


@router.get("/waterfall")
async def waterfall_endpoint(
    db: DbSession,
    f: FiltersDep,
    user: CurrentUser,
    metric: str = "revenue",
    dimension: str = "category",
    top_n: int = 8,
):
    f = _scoped_filters(f, user)
    if metric not in ALLOWED_METRICS_SCENARIO:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"invalid metric '{metric}'")
    if dimension not in ("region", "channel", "category", "product"):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"invalid dimension '{dimension}'")
    return await advanced.waterfall(db, f, metric=metric, dimension=dimension, top_n=top_n)


@router.get("/heatmap")
async def heatmap_endpoint(
    db: DbSession,
    f: FiltersDep,
    user: CurrentUser,
    metric: str = "revenue",
    row_dim: str = "region",
    col_dim: str = "category",
):
    f = _scoped_filters(f, user)
    if metric not in ALLOWED_METRICS_SCENARIO:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"invalid metric '{metric}'")
    return await advanced.heatmap(db, f, metric=metric, row_dim=row_dim, col_dim=col_dim)


@router.get("/scatter")
async def scatter_endpoint(
    db: DbSession,
    f: FiltersDep,
    user: CurrentUser,
    dimension: str = "product",
    x: str = "revenue",
    y: str = "margin_pct",
    size: str = "units",
):
    f = _scoped_filters(f, user)
    if dimension not in ("region", "channel", "category", "product"):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"invalid dimension '{dimension}'")
    return await advanced.scatter(db, f, dimension=dimension, x=x, y=y, size=size)


@router.get("/funnel")
async def funnel_endpoint(
    db: DbSession,
    f: FiltersDep,
    user: CurrentUser,
    metric: str = "revenue",
    dimension: str = "category",
    top_n: int = 8,
):
    f = _scoped_filters(f, user)
    if metric not in ALLOWED_METRICS_SCENARIO:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"invalid metric '{metric}'")
    return await advanced.funnel(db, f, metric=metric, dimension=dimension, top_n=top_n)


@router.get("/radar")
async def radar_endpoint(
    db: DbSession,
    f: FiltersDep,
    user: CurrentUser,
    dimension: str = "region",
    metrics: str = "revenue,orders,gross_margin,aov,units",
):
    f = _scoped_filters(f, user)
    if dimension not in ("region", "channel", "category", "product"):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"invalid dimension '{dimension}'")
    return await advanced.radar(db, f, dimension=dimension, metrics=metrics)


@router.get("/small-multiples")
async def small_multiples_endpoint(
    db: DbSession,
    f: FiltersDep,
    user: CurrentUser,
    metric: str = "revenue",
    dimension: str = "region",
    granularity: str = "month",
):
    f = _scoped_filters(f, user)
    if metric not in ALLOWED_METRICS_SCENARIO:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"invalid metric '{metric}'")
    if granularity not in ALLOWED_GRANULARITY:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"invalid granularity '{granularity}'")
    return await advanced.small_multiples(db, f, metric=metric, dimension=dimension, granularity=granularity)


@router.get("/key-influencers")
async def key_influencers_endpoint(db: DbSession, f: FiltersDep, user: CurrentUser, target: str = "revenue"):
    f = _scoped_filters(f, user)
    if target not in ALLOWED_METRICS_SCENARIO:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"invalid target '{target}'")
    return await key_influencers(db, f, target=target)


@router.get("/segmentation")
async def segmentation_endpoint(
    db: DbSession, f: FiltersDep, user: CurrentUser, dimension: str = "product", n_clusters: int = 4
):
    f = _scoped_filters(f, user)
    if dimension not in ("region", "channel", "category", "product"):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"invalid dimension '{dimension}'")
    return await segment(db, f, dimension=dimension, n_clusters=n_clusters)


@router.get("/forecast-scenarios")
async def forecast_scenarios_endpoint(
    db: DbSession,
    f: FiltersDep,
    user: CurrentUser,
    metric: str = "revenue",
    horizon: int = Query(30, ge=1, le=180),
    n_paths: int = Query(500, ge=50, le=2000),
    model: str | None = None,
):
    f = _scoped_filters(f, user)
    if metric not in ALLOWED_METRICS_SCENARIO:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"invalid metric '{metric}'")
    if model is not None and model not in fc.CANDIDATE_NAMES:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"invalid model '{model}' — must be one of {fc.CANDIDATE_NAMES}")
    series = await _daily_series(db, f, metric=metric)
    if len(series) < 7:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "insufficient history for probabilistic forecast (need ≥7 days)")
    # also enforce 180 for accurate bands? monte_carlo will raise
    import pandas as pd

    frame = pd.DataFrame(series)
    # convert ds to datetime
    frame["ds"] = pd.to_datetime(frame["ds"])
    frame["y"] = frame["y"].astype(float)
    # ensure continuous and sorted
    frame = frame.sort_values("ds").reset_index(drop=True)
    try:
        return monte_carlo(frame, horizon=horizon, n_paths=n_paths, model=model)
    except ValueError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e))


@router.get("/model-comparison")
async def model_comparison_endpoint(db: DbSession, f: FiltersDep, user: CurrentUser, metric: str = "revenue"):
    """Holdout MAPE for every candidate model on the selected series."""
    f = _scoped_filters(f, user)
    if metric not in ALLOWED_METRICS_SCENARIO:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"invalid metric '{metric}'")
    series = await _daily_series(db, f, metric=metric)
    if len(series) < fc.HOLDOUT_DAYS + 14:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "need at least ~104 days of history")
    import pandas as pd

    frame = pd.DataFrame(series)
    frame["ds"] = pd.to_datetime(frame["ds"])
    frame["y"] = frame["y"].astype(float)
    frame = frame.sort_values("ds").reset_index(drop=True)
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
