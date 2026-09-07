"""Analytics endpoints: /kpis, /sales, /finance, /inventory (Phase 3)."""

import dataclasses
from datetime import date, timedelta
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text

from app.api.deps import CurrentUser, DbSession, get_current_user, is_super_admin, require_role
from app.core.clock import business_today
from app.schemas.analytics import (
    DashboardOut,
    DataCoverage,
    DimensionRow,
    InventoryRow,
    KpiDefinitionOut,
    KpiDefinitionUpdate,
    KpiSummary,
    Paginated,
    PnlRow,
    Timeseries,
    TransactionRow,
)
from app.services.analytics import queries
from app.services.analytics.cache import get_query_cache
from app.services.analytics.diagnostics import diagnose_change
from app.services.analytics.queries import Filters


def get_filters(
    date_from: Annotated[date | None, Query(alias="from")] = None,
    date_to: Annotated[date | None, Query(alias="to")] = None,
    region: str | None = None,
    channel: str | None = None,
    category: str | None = None,
    regions: Annotated[str | None, Query()] = None,
    channels: Annotated[str | None, Query()] = None,
    categories: Annotated[str | None, Query()] = None,
) -> Filters:
    today = business_today()
    return Filters(
        date_from=date_from or (date_to or today) - timedelta(days=29),
        date_to=date_to or today,
        region=region,
        channel=channel,
        category=category,
        regions=tuple(r for r in (regions or "").split(",") if r),
        channels=tuple(c for c in (channels or "").split(",") if c),
        categories=tuple(c for c in (categories or "").split(",") if c),
    )


FiltersDep = Annotated[Filters, Depends(get_filters)]

router = APIRouter(tags=["analytics"], dependencies=[Depends(get_current_user)])


def _scoped_filters(f: Filters, user: CurrentUser) -> Filters:
    if is_super_admin(user):
        return f
    return dataclasses.replace(f, org_id=user.org_id)


@router.get("/kpis/summary", response_model=KpiSummary)
async def get_kpi_summary(db: DbSession, f: FiltersDep, user: CurrentUser) -> KpiSummary:
    f = _scoped_filters(f, user)
    cards = await queries.kpi_summary(db, f)
    return KpiSummary(period_start=f.date_from, period_end=f.date_to, cards=cards)


@router.get("/kpis/timeseries", response_model=Timeseries)
async def get_kpi_timeseries(
    db: DbSession,
    f: FiltersDep,
    user: CurrentUser,
    metric: Literal["revenue", "orders", "avg_order_value", "expense_total"] = "revenue",
    granularity: Literal["day", "week", "month", "quarter", "year"] = "day",
) -> Timeseries:
    f = _scoped_filters(f, user)
    points = await queries.kpi_timeseries(db, f, metric, granularity)
    return Timeseries(metric=metric, granularity=granularity, points=points)


@router.get("/kpis/definitions", response_model=list[KpiDefinitionOut])
async def list_kpi_definitions(
    db: DbSession,
    f: FiltersDep,
    user: CurrentUser,
) -> list[KpiDefinitionOut]:
    """Metadata-driven KPI registry: formula, unit, target, thresholds, visibility."""
    from sqlalchemy import select

    from app.models import KpiDefinition

    stmt = select(KpiDefinition).order_by(KpiDefinition.metric)
    if not is_super_admin(user) and user.org_id:
        from sqlalchemy import or_

        stmt = stmt.where(or_(KpiDefinition.org_id == user.org_id, KpiDefinition.org_id.is_(None)))
    rows = (await db.execute(stmt)).scalars().all()
    return [KpiDefinitionOut.model_validate(r) for r in rows]


@router.patch(
    "/kpis/definitions/{metric}",
    response_model=KpiDefinitionOut,
    dependencies=[Depends(require_role("admin"))],
)
async def update_kpi_definition(metric: str, body: KpiDefinitionUpdate, db: DbSession) -> KpiDefinitionOut:
    from fastapi import HTTPException
    from sqlalchemy import select

    from app.models import KpiDefinition

    definition = (await db.execute(select(KpiDefinition).where(KpiDefinition.metric == metric))).scalar_one_or_none()
    if definition is None:
        raise HTTPException(404, f"No KPI definition for '{metric}'")
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(definition, field, value)
    await db.commit()
    await db.refresh(definition)
    return KpiDefinitionOut.model_validate(definition)


@router.get("/sales/by-product", response_model=list[DimensionRow])
async def sales_by_product(db: DbSession, f: FiltersDep, user: CurrentUser) -> list[DimensionRow]:
    f = _scoped_filters(f, user)
    return await queries.sales_by_dimension(db, f, "product")


@router.get("/sales/by-category", response_model=list[DimensionRow])
async def sales_by_category(db: DbSession, f: FiltersDep, user: CurrentUser) -> list[DimensionRow]:
    f = _scoped_filters(f, user)
    return await queries.sales_by_dimension(db, f, "category")


@router.get("/sales/by-region", response_model=list[DimensionRow])
async def sales_by_region(db: DbSession, f: FiltersDep, user: CurrentUser) -> list[DimensionRow]:
    f = _scoped_filters(f, user)
    return await queries.sales_by_dimension(db, f, "region")


@router.get("/sales/by-channel", response_model=list[DimensionRow])
async def sales_by_channel(db: DbSession, f: FiltersDep, user: CurrentUser) -> list[DimensionRow]:
    f = _scoped_filters(f, user)
    return await queries.sales_by_dimension(db, f, "channel")


@router.get("/sales/transactions", response_model=Paginated[TransactionRow])
async def get_sales_transactions(
    db: DbSession,
    f: FiltersDep,
    user: CurrentUser,
    sku: str | None = None,
    search: str | None = None,
    sort_by: str | None = Query(
        None, description="Sort column: txn_date|product|channel|region|quantity|total_amount|ingested_at"
    ),
    sort_dir: str | None = Query(None, description="Sort direction: asc|desc"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
) -> Paginated[TransactionRow]:
    f = _scoped_filters(f, user)
    items, total = await queries.sales_transactions(db, f, page, page_size, sku, search, sort_by, sort_dir)
    from app.api.deps import redact_sensitive

    items = [redact_sensitive(user, item) for item in items]
    return Paginated(items=items, total=total, page=page, page_size=page_size)


@router.get("/finance/expenses-by-category", response_model=list[DimensionRow])
async def get_expenses_by_category(db: DbSession, f: FiltersDep, user: CurrentUser) -> list[DimensionRow]:
    f = _scoped_filters(f, user)
    return await queries.expenses_by_category(db, f)


@router.get(
    "/finance/pnl",
    response_model=list[PnlRow],
    dependencies=[Depends(require_role("manager"))],
)
async def get_pnl(db: DbSession, f: FiltersDep, user: CurrentUser) -> list[PnlRow]:
    f = _scoped_filters(f, user)
    return await queries.monthly_pnl(db, f)


@router.get("/inventory/levels", response_model=list[InventoryRow])
async def get_inventory_levels(
    db: DbSession,
    user: CurrentUser,
    below_reorder: bool = False,
    as_of: Annotated[date | None, Query(description="Newest snapshot on or before this date")] = None,
) -> list[InventoryRow]:
    org_id = None if is_super_admin(user) else user.org_id
    return await queries.inventory_levels(db, below_reorder_only=below_reorder, as_of=as_of, org_id=org_id)


@router.get("/data-coverage", response_model=DataCoverage)
async def get_data_coverage(db: DbSession, user: CurrentUser) -> DataCoverage:
    org_id = None if is_super_admin(user) else user.org_id
    return DataCoverage.model_validate(await queries.data_coverage(db, org_id=org_id))


@router.get("/watermark")
async def get_watermark(db: DbSession) -> dict:
    """Last ETL refresh watermark for the UI to show 'last updated' timestamp."""
    row = await db.execute(text("SELECT * FROM data_watermarks WHERE id = 1"))
    wm = row.first()
    if not wm:
        return {"last_refresh_at": None, "last_source": None, "last_trigger": None, "affected_range": None}
    return {
        "last_refresh_at": wm.last_refresh_at.isoformat() if wm.last_refresh_at else None,
        "last_source": wm.last_source,
        "last_trigger": wm.last_trigger,
        "affected_range": (
            {"start": wm.affected_range_start.isoformat(), "end": wm.affected_range_end.isoformat()}
            if wm.affected_range_start and wm.affected_range_end
            else None
        ),
        "details": wm.details,
    }


@router.get("/diagnostics/change")
async def diagnose_change_endpoint(
    db: DbSession,
    f: FiltersDep,
    user: CurrentUser,
    metric: Literal["revenue", "orders", "avg_order_value", "gross_margin", "expense_total"] = "revenue",
    dimensions: str = "region,channel,product",
) -> dict:
    f = _scoped_filters(f, user)
    dims = tuple(d.strip() for d in dimensions.split(",") if d.strip())
    return await diagnose_change(
        db,
        metric=metric,
        date_from=f.date_from,
        date_to=f.date_to,
        dimensions=dims,
        region=f.region,
        channel=f.channel,
        category=f.category,
        org_id=f.org_id,
    )


@router.get("/dashboard", response_model=DashboardOut)
async def get_dashboard(db: DbSession, f: FiltersDep, user: CurrentUser) -> DashboardOut:
    """Single-call dashboard — 1 DB session, 1 auth check, ~14 queries sequential.

    Replaces the 14 parallel GETs the overview fired on first paint
    (summary, 2× timeseries, by-channel/category/region, transactions,
    levels, anomalies, recommendations, forecasts, pnl). Sequential on one
    connection avoids the pool starvation that produced the 3–4s /me and
    500s. Result is cached 30s per org+range (same key as the individual
    cached queries). Partial failures are returned in `errors` instead of
    500ing the whole dashboard.
    """
    import logging

    from sqlalchemy import select as sa_select

    from app.models import Anomaly, Forecast

    logger = logging.getLogger(__name__)
    f = _scoped_filters(f, user)
    cache = get_query_cache()
    # Cache key is org + range (dashboard is range-driven)
    cache_key = cache._make_key(  # type: ignore[attr-defined]
        "dashboard",
        (str(f.org_id), f.date_from.isoformat(), f.date_to.isoformat(), f.region, f.channel, f.category),
        {},
    )
    cached = await cache.get(cache_key)
    if cached is not None:
        return DashboardOut(**cached)

    out: dict = {"errors": {}}
    # Use one session sequentially to avoid pool contention; each helper is
    # already @cached_query so second dashboard hit is instant.

    try:
        cards = await queries.kpi_summary(db, f)
        out["summary"] = {"period_start": f.date_from, "period_end": f.date_to, "cards": cards}
    except Exception as e:
        logger.warning("dashboard summary failed: %s", e, exc_info=True)
        out["errors"]["summary"] = str(e)

    try:
        out["timeseries_revenue"] = {
            "metric": "revenue",
            "granularity": "day",
            "points": await queries.kpi_timeseries(db, f, "revenue", "day"),
        }
    except Exception as e:
        out["errors"]["timeseries_revenue"] = str(e)

    try:
        out["timeseries_expense"] = {
            "metric": "expense_total",
            "granularity": "day",
            "points": await queries.kpi_timeseries(db, f, "expense_total", "day"),
        }
    except Exception as e:
        out["errors"]["timeseries_expense"] = str(e)

    for dim in ("channel", "category", "region"):
        try:
            out[f"by_{dim}"] = await queries.sales_by_dimension(db, f, dim)
        except Exception as e:
            out["errors"][f"by_{dim}"] = str(e)

    try:
        items, total = await queries.sales_transactions(db, f, 1, 6, None, None, None, None)
        out["transactions"] = {"items": items, "total": total, "page": 1, "page_size": 6}
    except Exception as e:
        out["errors"]["transactions"] = str(e)

    try:
        org_id = None if is_super_admin(user) else user.org_id
        out["levels"] = await queries.inventory_levels(db, below_reorder_only=True, org_id=org_id)
    except Exception as e:
        out["errors"]["levels"] = str(e)

    try:
        org_id = None if is_super_admin(user) else user.org_id
        stmt = sa_select(Anomaly).where(Anomaly.status == "open").order_by(Anomaly.detected_at.desc()).limit(50)
        if org_id is not None:
            stmt = stmt.where(Anomaly.org_id == org_id)
        rows = (await db.execute(stmt)).scalars().all()
        out["anomalies"] = [
            {
                "id": str(r.id),
                "metric": r.metric,
                "observed_value": str(r.observed_value),
                "expected_value": str(r.expected_value) if r.expected_value is not None else None,
                "severity": r.severity,
                "status": r.status,
                "context": r.context,
            }
            for r in rows
        ]
    except Exception as e:
        out["errors"]["anomalies"] = str(e)

    try:
        from app.services.ml.recommendations import generate_all_recommendations, scope_recommendations

        org_id = None if is_super_admin(user) else user.org_id
        recs = await generate_all_recommendations(db, org_id=org_id)
        recs = await scope_recommendations(db, recs, user)
        out["recommendations"] = recs[:10]
    except Exception as e:
        out["errors"]["recommendations"] = str(e)

    try:
        # Use the same logic as /forecasts but without re-implementing NaiveSeasonal
        # — call the service directly with a short horizon
        from app.api.v1.ml import _active_model

        org_id = None if is_super_admin(user) else user.org_id
        model = await _active_model(db, "revenue_daily", {}, org_id=org_id)
        if model is not None:
            rows = (
                (
                    await db.execute(
                        sa_select(Forecast)
                        .where(Forecast.model_id == model.id)
                        .order_by(Forecast.forecast_date)
                        .limit(30)
                    )
                )
                .scalars()
                .all()
            )
            out["forecasts"] = {
                "model_type": model.model_type,
                "model_version": model.version,
                "points": [{"forecast_date": r.forecast_date.isoformat(), "yhat": float(r.yhat)} for r in rows],
            }
    except Exception as e:
        out["errors"]["forecasts"] = str(e)

    try:
        # Only for manager/admin — otherwise skip to avoid 403 noise
        from app.api.deps import is_super_admin as _is_super

        if user.role in ("manager", "admin") or _is_super(user):
            out["pnl"] = await queries.monthly_pnl(db, f)
    except Exception as e:
        out["errors"]["pnl"] = str(e)

    if not out["errors"]:
        out.pop("errors", None)

    # Cache the successful shape for 30s; errors are cached too but for 10s
    # so a transient DB hiccup self-heals quickly.
    ttl = 30 if not out.get("errors") else 10
    await cache.set(cache_key, out, ttl)
    return DashboardOut(**out)


@router.get("/cache/stats")
async def cache_stats() -> dict:
    """Query cache hit/miss statistics."""
    from app.services.analytics.cache import get_cache_stats

    return await get_cache_stats()


@router.post("/cache/clear", dependencies=[Depends(require_role("admin"))])
async def cache_clear() -> dict:
    """Clear all cached queries."""
    from app.services.analytics.cache import clear_query_cache

    await clear_query_cache()
    return {"status": "cleared"}
