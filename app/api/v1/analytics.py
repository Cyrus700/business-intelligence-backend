"""Analytics endpoints: /kpis, /sales, /finance, /inventory (Phase 3)."""

import dataclasses
from datetime import date, timedelta
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text

from app.api.deps import CurrentUser, DbSession, get_current_user, is_super_admin, require_role
from app.core.clock import business_today
from app.schemas.analytics import (
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
