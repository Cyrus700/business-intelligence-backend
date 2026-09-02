"""Aggregate queries behind the analytics endpoints.

Headline KPIs read kpi_snapshots (the fast path) when no dimension filters are
set; dimension-filtered queries aggregate the fact tables directly using the
composite indexes (docs/03-database-schema.md).
"""

from dataclasses import dataclass
from datetime import date, timedelta
from uuid import UUID

from sqlalchemy import Date, and_, case, cast, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.clock import BUSINESS_TZ_NAME, business_today
from app.models import (
    Customer,
    Expense,
    InventoryLevel,
    KpiDefinition,
    Product,
    SalesTransaction,
)
from app.services.analytics.cache import cached_query

KPI_METRICS = ["revenue", "orders", "avg_order_value", "gross_margin", "expense_total"]


@dataclass(frozen=True)
class Filters:
    date_from: date
    date_to: date
    region: str | None = None
    channel: str | None = None
    category: str | None = None
    regions: tuple[str, ...] = ()
    channels: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    org_id: UUID | None = None

    @property
    def has_dimensions(self) -> bool:
        return any(
            (self.region, self.channel, self.category, self.regions, self.channels, self.categories)
        )

    def previous_period(self) -> tuple[date, date]:
        span = (self.date_to - self.date_from).days + 1
        return self.date_from - timedelta(days=span), self.date_from - timedelta(days=1)


def _sales_conditions(f: Filters, date_from: date, date_to: date) -> list:
    conditions = [SalesTransaction.txn_date.between(date_from, date_to)]
    if f.org_id is not None:
        conditions.append(SalesTransaction.org_id == f.org_id)
    if f.regions:
        conditions.append(SalesTransaction.region.in_(f.regions))
    elif f.region:
        conditions.append(SalesTransaction.region == f.region)
    if f.channels:
        conditions.append(SalesTransaction.channel.in_(f.channels))
    elif f.channel:
        conditions.append(SalesTransaction.channel == f.channel)
    if f.categories:
        cat_filter = Product.category.in_(f.categories)
        if f.org_id is not None:
            cat_filter = (Product.category.in_(f.categories)) & (Product.org_id == f.org_id)
        conditions.append(
            SalesTransaction.product_id.in_(
                select(Product.id).where(cat_filter)
            )
        )
    elif f.category:
        cat_filter = Product.category == f.category
        if f.org_id is not None:
            cat_filter = (Product.category == f.category) & (Product.org_id == f.org_id)
        conditions.append(
            SalesTransaction.product_id.in_(
                select(Product.id).where(cat_filter)
            )
        )
    return conditions


def _org_filter(col, org_id: UUID | None) -> list:
    return [col == org_id] if org_id is not None else []


@cached_query(ttl_seconds=30)
async def _sales_kpis(
    db: AsyncSession, f: Filters, date_from: date, date_to: date
) -> dict[str, float]:
    stmt = (
        select(
            func.coalesce(func.sum(SalesTransaction.total_amount), 0).label("revenue"),
            func.count(SalesTransaction.id).label("orders"),
            func.coalesce(func.avg(SalesTransaction.total_amount), 0).label("avg_order_value"),
            func.coalesce(
                func.sum(
                    SalesTransaction.total_amount
                    - func.coalesce(Product.unit_cost, 0) * SalesTransaction.quantity
                ),
                0,
            ).label("gross_margin"),
        )
        .select_from(
            SalesTransaction.__table__.outerjoin(
                Product.__table__, Product.id == SalesTransaction.product_id
            )
        )
        .where(and_(*_sales_conditions(f, date_from, date_to)))
    )
    row = (await db.execute(stmt)).one()
    return {
        "revenue": float(row.revenue),
        "orders": float(row.orders),
        "avg_order_value": float(row.avg_order_value),
        "gross_margin": float(row.gross_margin),
    }


@cached_query(ttl_seconds=30)
async def _expense_kpi(db: AsyncSession, date_from: date, date_to: date, org_id: UUID | None = None) -> float:
    conditions = [Expense.expense_date.between(date_from, date_to)]
    if org_id is not None:
        conditions.append(Expense.org_id == org_id)
    stmt = select(func.coalesce(func.sum(Expense.amount), 0)).where(and_(*conditions))
    return float((await db.execute(stmt)).scalar_one())


async def kpi_summary(db: AsyncSession, f: Filters) -> list[dict]:
    current = await _sales_kpis(db, f, f.date_from, f.date_to)
    prev_from, prev_to = f.previous_period()
    previous = await _sales_kpis(db, f, prev_from, prev_to)
    if not f.has_dimensions:  # expenses have no sales dimensions
        current["expense_total"] = await _expense_kpi(db, f.date_from, f.date_to, f.org_id)
        previous["expense_total"] = await _expense_kpi(db, prev_from, prev_to, f.org_id)

    # Metadata-driven parameters (Phase 5): labels, units, targets and
    # thresholds come from kpi_definitions, editable by admins.
    def_stmt = select(KpiDefinition).where(KpiDefinition.is_active.is_(True))
    if f.org_id is not None:
        def_stmt = def_stmt.where(
            or_(KpiDefinition.org_id == f.org_id, KpiDefinition.org_id.is_(None))
        )
    definitions = {
        d.metric: d for d in (await db.execute(def_stmt)).scalars()
    }

    cards = []
    for metric, value in current.items():
        prev = previous.get(metric)
        change = round((value - prev) / prev * 100, 1) if prev else None
        definition = definitions.get(metric)
        card = {
            "metric": metric,
            "value": round(value, 2),
            "previous_value": round(prev, 2) if prev is not None else None,
            "change_pct": change,
        }
        if definition is not None:
            target = float(definition.target_value) if definition.target_value is not None else None
            threshold = (
                float(definition.threshold_low) if definition.threshold_low is not None else None
            )
            card["label"] = definition.label
            card["unit"] = definition.unit
            card["target_value"] = target
            status = None
            if target is not None:
                gap = value / target if target else None
                if definition.higher_is_better:
                    status = (
                        "off_target"
                        if gap and gap < 0.8
                        else ("near_target" if gap and gap < 1.0 else "on_track")
                    )
                else:
                    status = "on_track" if value <= max(target, threshold or 0) else "near_target"
                card["achievement_pct"] = round(gap * 100, 1) if gap is not None else None
            elif threshold is not None:
                if definition.higher_is_better:
                    status = "off_target" if value < threshold else "on_track"
                else:
                    status = "on_track" if value <= threshold else "off_target"
            card["status"] = status
        cards.append(card)
    return cards


async def kpi_timeseries(db: AsyncSession, f: Filters, metric: str, granularity: str) -> list[dict]:
    if metric == "expense_total":
        bucket = cast(func.date_trunc(granularity, cast(Expense.expense_date, Date)), Date)
        base_cond = [Expense.expense_date.between(f.date_from, f.date_to)]
        if f.org_id is not None:
            base_cond.append(Expense.org_id == f.org_id)
        stmt = (
            select(bucket.label("period"), func.sum(Expense.amount).label("value"))
            .where(and_(*base_cond))
            .group_by(bucket)
            .order_by(bucket)
        )
    else:
        value_expr = {
            "revenue": func.sum(SalesTransaction.total_amount),
            "orders": func.count(SalesTransaction.id),
            "avg_order_value": func.avg(SalesTransaction.total_amount),
        }[metric]
        bucket = cast(
            func.date_trunc(granularity, cast(SalesTransaction.txn_date, Date)), Date
        )
        stmt = (
            select(bucket.label("period"), value_expr.label("value"))
            .where(and_(*_sales_conditions(f, f.date_from, f.date_to)))
            .group_by(bucket)
            .order_by(bucket)
        )
    rows = (await db.execute(stmt)).all()
    return [{"period": r.period, "value": round(float(r.value), 2)} for r in rows]


async def sales_by_dimension(db: AsyncSession, f: Filters, dimension: str) -> list[dict]:
    # Org filter: sales rows already filtered via _sales_conditions, but join must not leak cross-org products
    if dimension == "product":
        key, sku = Product.name, Product.sku
        join_cond = Product.id == SalesTransaction.product_id
        if f.org_id is not None:
            join_cond = (Product.id == SalesTransaction.product_id) & (Product.org_id == f.org_id)
        stmt_from = SalesTransaction.__table__.join(Product.__table__, join_cond)
        group = [Product.name, Product.sku]
    elif dimension == "category":
        join_cond = Product.id == SalesTransaction.product_id
        if f.org_id is not None:
            join_cond = (Product.id == SalesTransaction.product_id) & (Product.org_id == f.org_id)
        key, sku = Product.category, None
        stmt_from = SalesTransaction.__table__.join(Product.__table__, join_cond)
        group = [Product.category]
    else:
        key = getattr(SalesTransaction, dimension)  # region | channel
        sku, stmt_from, group = None, SalesTransaction.__table__, [key]

    cols = [
        key.label("key"),
        func.count(SalesTransaction.id).label("orders"),
        func.sum(SalesTransaction.quantity).label("quantity"),
        func.sum(SalesTransaction.total_amount).label("revenue"),
    ]
    if sku is not None:
        cols.append(sku.label("sku"))
    stmt = (
        select(*cols)
        .select_from(stmt_from)
        .where(and_(*_sales_conditions(f, f.date_from, f.date_to)))
        .group_by(*group)
        .order_by(func.sum(SalesTransaction.total_amount).desc())
    )
    rows = (await db.execute(stmt)).all()
    total = sum(float(r.revenue) for r in rows) or 1.0
    return [
        {
            "key": r.key or "(unknown)",
            "sku": getattr(r, "sku", None),
            "orders": r.orders,
            "quantity": int(r.quantity or 0),
            "revenue": round(float(r.revenue), 2),
            "share_pct": round(float(r.revenue) / total * 100, 1),
        }
        for r in rows
    ]


async def sales_transactions(
    db: AsyncSession,
    f: Filters,
    page: int,
    page_size: int,
    sku: str | None = None,
    search: str | None = None,
    sort_by: str | None = None,
    sort_dir: str | None = None,
) -> tuple[list[dict], int]:
    conditions = _sales_conditions(f, f.date_from, f.date_to)
    if sku:
        conditions.append(Product.sku == sku)
    if search:
        q = f"%{search}%"
        conditions.append(
            or_(Product.name.ilike(q), Customer.name.ilike(q), SalesTransaction.channel.ilike(q))
        )
    base = (
        select(
            SalesTransaction.id,
            SalesTransaction.txn_date,
            Product.name.label("product"),
            Product.sku,
            Customer.name.label("customer"),
            SalesTransaction.channel,
            SalesTransaction.region,
            SalesTransaction.quantity,
            SalesTransaction.unit_price,
            SalesTransaction.discount,
            SalesTransaction.total_amount,
            SalesTransaction.ingested_at,
            SalesTransaction.etl_job_id,
            SalesTransaction.source_id,
        )
        .select_from(
            SalesTransaction.__table__.outerjoin(
                Product.__table__, Product.id == SalesTransaction.product_id
            ).outerjoin(Customer.__table__, Customer.id == SalesTransaction.customer_id)
        )
        .where(and_(*conditions))
    )
    total = (await db.execute(select(func.count()).select_from(base.subquery()))).scalar_one()

    # — Professional sorting: whitelisted columns only (prevents injection)
    sort_map = {
        "txn_date": SalesTransaction.txn_date,
        "product": Product.name,
        "customer": Customer.name,
        "channel": SalesTransaction.channel,
        "region": SalesTransaction.region,
        "quantity": SalesTransaction.quantity,
        "total_amount": SalesTransaction.total_amount,
        "ingested_at": SalesTransaction.ingested_at,
    }
    col = sort_map.get((sort_by or "").lower(), SalesTransaction.txn_date)
    desc = (sort_dir or "desc").lower() == "desc"
    order = col.desc() if desc else col.asc()
    # Tie-breaker for stable pagination
    order2 = SalesTransaction.id.desc() if desc else SalesTransaction.id.asc()

    rows = (
        await db.execute(base.order_by(order, order2).offset((page - 1) * page_size).limit(page_size))
    ).all()
    items = [
        {
            **dict(r._mapping),
            "unit_price": float(r.unit_price),
            "discount": float(r.discount),
            "total_amount": float(r.total_amount),
            "ingested_at": r.ingested_at.isoformat() if r.ingested_at else None,
            "etl_job_id": str(r.etl_job_id) if r.etl_job_id else None,
            "source_id": str(r.source_id) if r.source_id else None,
        }
        for r in rows
    ]
    return items, total


async def expenses_by_category(db: AsyncSession, f: Filters) -> list[dict]:
    conds = [Expense.expense_date.between(f.date_from, f.date_to)]
    if f.org_id is not None:
        conds.append(Expense.org_id == f.org_id)
    stmt = (
        select(
            Expense.category.label("key"),
            func.count(Expense.id).label("orders"),
            func.sum(Expense.amount).label("revenue"),
        )
        .where(and_(*conds))
        .group_by(Expense.category)
        .order_by(func.sum(Expense.amount).desc())
    )
    rows = (await db.execute(stmt)).all()
    total = sum(float(r.revenue) for r in rows) or 1.0
    return [
        {
            "key": r.key,
            "orders": r.orders,
            "revenue": round(float(r.revenue), 2),
            "share_pct": round(float(r.revenue) / total * 100, 1),
        }
        for r in rows
    ]


async def monthly_pnl(db: AsyncSession, f: Filters) -> list[dict]:
    sales_month = cast(func.date_trunc("month", cast(SalesTransaction.txn_date, Date)), Date)
    sales_conds = [SalesTransaction.txn_date.between(f.date_from, f.date_to)]
    if f.org_id is not None:
        sales_conds.append(SalesTransaction.org_id == f.org_id)
    revenue_q = (
        select(
            sales_month.label("month"),
            func.sum(SalesTransaction.total_amount).label("revenue"),
            func.sum(
                SalesTransaction.total_amount
                - func.coalesce(Product.unit_cost, 0) * SalesTransaction.quantity
            ).label("gross_margin"),
        )
        .select_from(
            SalesTransaction.__table__.outerjoin(
                Product.__table__, Product.id == SalesTransaction.product_id
            )
        )
        .where(and_(*sales_conds))
        .group_by(sales_month)
        .subquery()
    )
    expense_month = cast(func.date_trunc("month", cast(Expense.expense_date, Date)), Date)
    exp_conds = [Expense.expense_date.between(f.date_from, f.date_to)]
    if f.org_id is not None:
        exp_conds.append(Expense.org_id == f.org_id)
    expense_q = (
        select(expense_month.label("month"), func.sum(Expense.amount).label("expenses"))
        .where(and_(*exp_conds))
        .group_by(expense_month)
        .subquery()
    )
    stmt = (
        select(
            func.coalesce(revenue_q.c.month, expense_q.c.month).label("month"),
            func.coalesce(revenue_q.c.revenue, 0).label("revenue"),
            func.coalesce(revenue_q.c.gross_margin, 0).label("gross_margin"),
            func.coalesce(expense_q.c.expenses, 0).label("expenses"),
        )
        .select_from(
            revenue_q.outerjoin(expense_q, revenue_q.c.month == expense_q.c.month, full=True)
        )
        .order_by("month")
    )
    rows = (await db.execute(stmt)).all()
    return [
        {
            "month": r.month,
            "revenue": round(float(r.revenue), 2),
            "expenses": round(float(r.expenses), 2),
            "gross_margin": round(float(r.gross_margin), 2),
            "net": round(float(r.gross_margin) - float(r.expenses), 2),
        }
        for r in rows
    ]


@cached_query(ttl_seconds=60)
async def data_coverage(db: AsyncSession, org_id=None) -> dict:
    """What the warehouse actually holds, per fact table.

    Grounds every "what about <date>?" answer: without it the assistant cannot
    tell "that day had zero sales" apart from "that day is outside the data we
    loaded", and will confidently report 0 for both. Also surfaces the newest
    ingestion timestamp so the UI can show data freshness.
    """
    def _org_cond(col, oid):
        return [col == oid] if oid is not None else []

    sales = (
        await db.execute(
            select(
                func.min(SalesTransaction.txn_date),
                func.max(SalesTransaction.txn_date),
                func.count(SalesTransaction.id),
                func.max(SalesTransaction.ingested_at),
            ).where(and_(*_org_cond(SalesTransaction.org_id, org_id)))
        )
    ).one()
    expenses = (
        await db.execute(
            select(
                func.min(Expense.expense_date),
                func.max(Expense.expense_date),
                func.count(Expense.id),
                func.max(Expense.ingested_at),
            ).where(and_(*_org_cond(Expense.org_id, org_id)))
        )
    ).one()
    inventory = (
        await db.execute(
            select(
                func.min(InventoryLevel.snapshot_date),
                func.max(InventoryLevel.snapshot_date),
                func.count(InventoryLevel.id),
                func.max(InventoryLevel.ingested_at),
            ).where(and_(*_org_cond(InventoryLevel.org_id, org_id)))
        )
    ).one()

    def block(row) -> dict:
        return {
            "first_date": row[0],
            "last_date": row[1],
            "row_count": int(row[2] or 0),
            "last_ingested_at": row[3],
        }

    blocks = {
        "sales": block(sales),
        "expenses": block(expenses),
        "inventory": block(inventory),
    }
    firsts = [b["first_date"] for b in blocks.values() if b["first_date"]]
    lasts = [b["last_date"] for b in blocks.values() if b["last_date"]]
    ingests = [b["last_ingested_at"] for b in blocks.values() if b["last_ingested_at"]]
    today = business_today()
    last_overall = max(lasts) if lasts else None
    return {
        **blocks,
        "first_date": min(firsts) if firsts else None,
        "last_date": last_overall,
        "last_ingested_at": max(ingests) if ingests else None,
        "today": today,
        "timezone": BUSINESS_TZ_NAME,
        # How far behind "today" the newest business day is — 0 means the
        # warehouse has data for today, so a "Today" filter is meaningful.
        "days_behind": (today - last_overall).days if last_overall else None,
    }


async def inventory_levels(
    db: AsyncSession, below_reorder_only: bool = False, as_of: date | None = None, org_id=None
) -> list[dict]:
    """Stock position per product.

    Inventory is a snapshot series, not a period aggregate, so a date range does
    not apply: the answer is "the newest snapshot on or before ``as_of``".
    ``as_of=None`` means the newest snapshot overall.
    """
    latest_q = select(
        InventoryLevel.product_id,
        func.max(InventoryLevel.snapshot_date).label("latest_date"),
    )
    if org_id is not None:
        latest_q = latest_q.where(InventoryLevel.org_id == org_id)
    if as_of is not None:
        latest_q = latest_q.where(InventoryLevel.snapshot_date <= as_of)
    latest = latest_q.group_by(InventoryLevel.product_id).subquery()
    join_cond = Product.id == InventoryLevel.product_id
    if org_id is not None:
        join_cond = (Product.id == InventoryLevel.product_id) & (Product.org_id == org_id)
    base_select = (
        select(
            Product.sku,
            Product.name.label("product"),
            Product.category,
            InventoryLevel.snapshot_date,
            InventoryLevel.quantity_on_hand,
            InventoryLevel.reorder_level,
            InventoryLevel.warehouse,
            case(
                (InventoryLevel.quantity_on_hand <= InventoryLevel.reorder_level, True),
                else_=False,
            ).label("below_reorder"),
        )
        .select_from(
            InventoryLevel.__table__.join(
                latest,
                and_(
                    latest.c.product_id == InventoryLevel.product_id,
                    latest.c.latest_date == InventoryLevel.snapshot_date,
                ),
            ).join(Product.__table__, join_cond)
        )
        .order_by(Product.sku)
    )
    if org_id is not None:
        base_select = base_select.where(InventoryLevel.org_id == org_id)
    rows = [dict(r._mapping) for r in (await db.execute(base_select)).all()]
    if below_reorder_only:
        rows = [r for r in rows if r["below_reorder"]]
    return rows
