"""Analytics endpoints: /kpis, /sales, /finance, /inventory (Phase 3)."""

from datetime import date, timedelta
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query

from app.api.deps import DbSession, get_current_user, require_role
from app.core.clock import business_today
from app.schemas.analytics import (
    DataCoverage,
    DimensionRow,
    InventoryRow,
    KpiSummary,
    Paginated,
    PnlRow,
    Timeseries,
    TransactionRow,
)
from app.services.analytics import queries
from app.services.analytics.queries import Filters


def get_filters(
    date_from: Annotated[date | None, Query(alias="from")] = None,
    date_to: Annotated[date | None, Query(alias="to")] = None,
    region: str | None = None,
    channel: str | None = None,
    category: str | None = None,
) -> Filters:
    today = business_today()
    return Filters(
        date_from=date_from or (date_to or today) - timedelta(days=29),
        date_to=date_to or today,
        region=region,
        channel=channel,
        category=category,
    )


FiltersDep = Annotated[Filters, Depends(get_filters)]

router = APIRouter(tags=["analytics"], dependencies=[Depends(get_current_user)])


@router.get("/kpis/summary", response_model=KpiSummary)
async def get_kpi_summary(db: DbSession, f: FiltersDep) -> KpiSummary:
    cards = await queries.kpi_summary(db, f)
    return KpiSummary(period_start=f.date_from, period_end=f.date_to, cards=cards)


@router.get("/kpis/timeseries", response_model=Timeseries)
async def get_kpi_timeseries(
    db: DbSession,
    f: FiltersDep,
    metric: Literal["revenue", "orders", "avg_order_value", "expense_total"] = "revenue",
    granularity: Literal["day", "week", "month"] = "day",
) -> Timeseries:
    points = await queries.kpi_timeseries(db, f, metric, granularity)
    return Timeseries(metric=metric, granularity=granularity, points=points)


@router.get("/sales/by-product", response_model=list[DimensionRow])
async def sales_by_product(db: DbSession, f: FiltersDep) -> list[DimensionRow]:
    return await queries.sales_by_dimension(db, f, "product")


@router.get("/sales/by-category", response_model=list[DimensionRow])
async def sales_by_category(db: DbSession, f: FiltersDep) -> list[DimensionRow]:
    return await queries.sales_by_dimension(db, f, "category")


@router.get("/sales/by-region", response_model=list[DimensionRow])
async def sales_by_region(db: DbSession, f: FiltersDep) -> list[DimensionRow]:
    return await queries.sales_by_dimension(db, f, "region")


@router.get("/sales/by-channel", response_model=list[DimensionRow])
async def sales_by_channel(db: DbSession, f: FiltersDep) -> list[DimensionRow]:
    return await queries.sales_by_dimension(db, f, "channel")


@router.get("/sales/transactions", response_model=Paginated[TransactionRow])
async def get_sales_transactions(
    db: DbSession,
    f: FiltersDep,
    user: Annotated[object, Depends(get_current_user)],
    sku: str | None = None,
    search: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
) -> Paginated[TransactionRow]:
    items, total = await queries.sales_transactions(db, f, page, page_size, sku, search)
    # field-level redaction: analysts see line-level rows without the money
    # fields (aggregates stay available via the dimension views)
    from app.api.deps import redact_sensitive

    items = [redact_sensitive(user, item) for item in items]
    return Paginated(items=items, total=total, page=page, page_size=page_size)


@router.get("/finance/expenses-by-category", response_model=list[DimensionRow])
async def get_expenses_by_category(db: DbSession, f: FiltersDep) -> list[DimensionRow]:
    return await queries.expenses_by_category(db, f)


@router.get(
    "/finance/pnl",
    response_model=list[PnlRow],
    dependencies=[Depends(require_role("manager"))],
)
async def get_pnl(db: DbSession, f: FiltersDep) -> list[PnlRow]:
    return await queries.monthly_pnl(db, f)


@router.get("/inventory/levels", response_model=list[InventoryRow])
async def get_inventory_levels(
    db: DbSession,
    below_reorder: bool = False,
    as_of: Annotated[
        date | None, Query(description="Newest snapshot on or before this date")
    ] = None,
) -> list[InventoryRow]:
    return await queries.inventory_levels(db, below_reorder_only=below_reorder, as_of=as_of)


@router.get("/data-coverage", response_model=DataCoverage)
async def get_data_coverage(db: DbSession) -> DataCoverage:
    """Which dates the warehouse actually holds, and when it was last loaded.

    Lets the UI label a range picker honestly ("no data for today — latest is
    X") instead of rendering an empty chart that looks like a zero.
    """
    return DataCoverage.model_validate(await queries.data_coverage(db))
