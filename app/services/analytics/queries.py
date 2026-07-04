"""Aggregate queries behind the analytics endpoints.

Headline KPIs read kpi_snapshots (the fast path) when no dimension filters are
set; dimension-filtered queries aggregate the fact tables directly using the
composite indexes (docs/03-database-schema.md).
"""

from dataclasses import dataclass
from datetime import date, timedelta

from sqlalchemy import Date, and_, case, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    Customer,
    Expense,
    InventoryLevel,
    KpiSnapshot,
    Product,
    SalesTransaction,
)

KPI_METRICS = ["revenue", "orders", "avg_order_value", "gross_margin", "expense_total"]


@dataclass(frozen=True)
class Filters:
    date_from: date
    date_to: date
    region: str | None = None
    channel: str | None = None
    category: str | None = None

    @property
    def has_dimensions(self) -> bool:
        return any((self.region, self.channel, self.category))

    def previous_period(self) -> tuple[date, date]:
        span = (self.date_to - self.date_from).days + 1
        return self.date_from - timedelta(days=span), self.date_from - timedelta(days=1)


def _sales_conditions(f: Filters, date_from: date, date_to: date) -> list:
    conditions = [SalesTransaction.txn_date.between(date_from, date_to)]
    if f.region:
        conditions.append(SalesTransaction.region == f.region)
    if f.channel:
        conditions.append(SalesTransaction.channel == f.channel)
    if f.category:
        conditions.append(
            SalesTransaction.product_id.in_(
                select(Product.id).where(Product.category == f.category)
            )
        )
    return conditions


async def _sales_kpis(
    db: AsyncSession, f: Filters, date_from: date, date_to: date
) -> dict[str, float]:
    stmt = select(
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
    ).select_from(
        SalesTransaction.__table__.outerjoin(
            Product.__table__, Product.id == SalesTransaction.product_id
        )
    ).where(and_(*_sales_conditions(f, date_from, date_to)))
    row = (await db.execute(stmt)).one()
    return {
        "revenue": float(row.revenue),
        "orders": float(row.orders),
        "avg_order_value": float(row.avg_order_value),
        "gross_margin": float(row.gross_margin),
    }


async def _expense_kpi(db: AsyncSession, date_from: date, date_to: date) -> float:
    stmt = select(func.coalesce(func.sum(Expense.amount), 0)).where(
        Expense.expense_date.between(date_from, date_to)
    )
    return float((await db.execute(stmt)).scalar_one())


async def kpi_summary(db: AsyncSession, f: Filters) -> list[dict]:
    current = await _sales_kpis(db, f, f.date_from, f.date_to)
    prev_from, prev_to = f.previous_period()
    previous = await _sales_kpis(db, f, prev_from, prev_to)
    if not f.has_dimensions:  # expenses have no sales dimensions
        current["expense_total"] = await _expense_kpi(db, f.date_from, f.date_to)
        previous["expense_total"] = await _expense_kpi(db, prev_from, prev_to)

    cards = []
    for metric, value in current.items():
        prev = previous.get(metric)
        change = round((value - prev) / prev * 100, 1) if prev else None
        cards.append(
            {"metric": metric, "value": round(value, 2),
             "previous_value": round(prev, 2) if prev is not None else None,
             "change_pct": change}
        )
    return cards


async def kpi_timeseries(
    db: AsyncSession, f: Filters, metric: str, granularity: str
) -> list[dict]:
    if metric == "expense_total":
        bucket = func.date_trunc(granularity, cast(Expense.expense_date, Date))
        stmt = (
            select(bucket.label("period"), func.sum(Expense.amount).label("value"))
            .where(Expense.expense_date.between(f.date_from, f.date_to))
            .group_by(bucket)
            .order_by(bucket)
        )
    else:
        value_expr = {
            "revenue": func.sum(SalesTransaction.total_amount),
            "orders": func.count(SalesTransaction.id),
            "avg_order_value": func.avg(SalesTransaction.total_amount),
        }[metric]
        bucket = func.date_trunc(granularity, cast(SalesTransaction.txn_date, Date))
        stmt = (
            select(bucket.label("period"), value_expr.label("value"))
            .where(and_(*_sales_conditions(f, f.date_from, f.date_to)))
            .group_by(bucket)
            .order_by(bucket)
        )
    rows = (await db.execute(stmt)).all()
    return [{"period": r.period.date(), "value": round(float(r.value), 2)} for r in rows]


async def sales_by_dimension(db: AsyncSession, f: Filters, dimension: str) -> list[dict]:
    if dimension == "product":
        key, sku = Product.name, Product.sku
        stmt_from = SalesTransaction.__table__.join(
            Product.__table__, Product.id == SalesTransaction.product_id
        )
        group = [Product.name, Product.sku]
    elif dimension == "category":
        key, sku = Product.category, None
        stmt_from = SalesTransaction.__table__.join(
            Product.__table__, Product.id == SalesTransaction.product_id
        )
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
    db: AsyncSession, f: Filters, page: int, page_size: int, sku: str | None = None
) -> tuple[list[dict], int]:
    conditions = _sales_conditions(f, f.date_from, f.date_to)
    if sku:
        conditions.append(Product.sku == sku)
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
        )
        .select_from(
            SalesTransaction.__table__.outerjoin(
                Product.__table__, Product.id == SalesTransaction.product_id
            ).outerjoin(Customer.__table__, Customer.id == SalesTransaction.customer_id)
        )
        .where(and_(*conditions))
    )
    total = (
        await db.execute(select(func.count()).select_from(base.subquery()))
    ).scalar_one()
    rows = (
        await db.execute(
            base.order_by(SalesTransaction.txn_date.desc(), SalesTransaction.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
    ).all()
    items = [
        {
            **dict(r._mapping),
            "unit_price": float(r.unit_price),
            "discount": float(r.discount),
            "total_amount": float(r.total_amount),
        }
        for r in rows
    ]
    return items, total


async def expenses_by_category(db: AsyncSession, f: Filters) -> list[dict]:
    stmt = (
        select(
            Expense.category.label("key"),
            func.count(Expense.id).label("orders"),
            func.sum(Expense.amount).label("revenue"),
        )
        .where(Expense.expense_date.between(f.date_from, f.date_to))
        .group_by(Expense.category)
        .order_by(func.sum(Expense.amount).desc())
    )
    rows = (await db.execute(stmt)).all()
    total = sum(float(r.revenue) for r in rows) or 1.0
    return [
        {"key": r.key, "orders": r.orders, "revenue": round(float(r.revenue), 2),
         "share_pct": round(float(r.revenue) / total * 100, 1)}
        for r in rows
    ]


async def monthly_pnl(db: AsyncSession, f: Filters) -> list[dict]:
    sales_month = func.date_trunc("month", cast(SalesTransaction.txn_date, Date))
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
        .where(SalesTransaction.txn_date.between(f.date_from, f.date_to))
        .group_by(sales_month)
        .subquery()
    )
    expense_month = func.date_trunc("month", cast(Expense.expense_date, Date))
    expense_q = (
        select(expense_month.label("month"), func.sum(Expense.amount).label("expenses"))
        .where(Expense.expense_date.between(f.date_from, f.date_to))
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
            "month": r.month.date(),
            "revenue": round(float(r.revenue), 2),
            "expenses": round(float(r.expenses), 2),
            "gross_margin": round(float(r.gross_margin), 2),
            "net": round(float(r.gross_margin) - float(r.expenses), 2),
        }
        for r in rows
    ]


async def inventory_levels(db: AsyncSession, below_reorder_only: bool = False) -> list[dict]:
    latest = (
        select(
            InventoryLevel.product_id,
            func.max(InventoryLevel.snapshot_date).label("latest_date"),
        )
        .group_by(InventoryLevel.product_id)
        .subquery()
    )
    stmt = (
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
            ).join(Product.__table__, Product.id == InventoryLevel.product_id)
        )
        .order_by(Product.sku)
    )
    rows = [dict(r._mapping) for r in (await db.execute(stmt)).all()]
    if below_reorder_only:
        rows = [r for r in rows if r["below_reorder"]]
    return rows
