"""Loaders: canonical records → warehouse tables, idempotently.

Sales/expenses upsert on row_hash (ON CONFLICT DO NOTHING), so re-running the
same file can never double-count. Inventory upserts on its natural key
(snapshot_date, product_id, warehouse) taking the latest value.
"""

import uuid
from collections.abc import Iterator
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.clock import business_now
from app.models import Customer, Expense, InventoryLevel, Product, SalesTransaction
from app.services.etl.base import LoadResult

# asyncpg caps bind parameters at 32767 per statement; widest row is 12 params,
# so 2000 rows/batch stays comfortably under the limit.
BATCH_SIZE = 2000


def _batches(rows: list[dict[str, Any]]) -> Iterator[list[dict[str, Any]]]:
    for i in range(0, len(rows), BATCH_SIZE):
        yield rows[i : i + BATCH_SIZE]


async def _ensure_products(db: AsyncSession, records: list[dict[str, Any]], org_id: uuid.UUID | None = None) -> dict[str, uuid.UUID]:
    wanted = {r["sku"]: r for r in records if r.get("sku")}
    if not wanted:
        return {}
    stmt = select(Product).where(Product.sku.in_(wanted))
    if org_id is not None:
        stmt = stmt.where(Product.org_id == org_id)
    existing = {
        p.sku: p.id for p in (await db.execute(stmt)).scalars()
    }
    for sku, r in wanted.items():
        if sku not in existing:
            product = Product(
                sku=sku,
                name=r.get("product_name") or sku,
                category=r.get("category"),
                unit_price=r.get("unit_price"),
                org_id=org_id,
            )
            db.add(product)
            await db.flush()
            existing[sku] = product.id
    return existing


async def _ensure_customers(
    db: AsyncSession, records: list[dict[str, Any]]
) -> dict[str, uuid.UUID]:
    wanted = {r["customer_name"]: r for r in records if r.get("customer_name")}
    if not wanted:
        return {}
    existing = {
        c.name: c.id
        for c in (await db.execute(select(Customer).where(Customer.name.in_(wanted)))).scalars()
    }
    for name, r in wanted.items():
        if name not in existing:
            customer = Customer(
                name=name, segment=r.get("segment"), city=r.get("city"), region=r.get("region")
            )
            db.add(customer)
            await db.flush()
            existing[name] = customer.id
    return existing


async def load_sales(
    db: AsyncSession,
    records: list[dict[str, Any]],
    source_id: uuid.UUID | None,
    etl_job_id: uuid.UUID | None,
    org_id: uuid.UUID | None = None,
) -> LoadResult:
    if not records:
        return LoadResult()
    products = await _ensure_products(db, records, org_id=org_id)
    customers = await _ensure_customers(db, records)
    # One stamp for the whole run: every row from an upload shares its
    # ingestion instant, so "uploaded today" groups cleanly.
    ingested_at = business_now().replace(tzinfo=None)
    rows = [
        {
            "txn_date": r["txn_date"],
            "product_id": products.get(r["sku"]),
            "customer_id": customers.get(r.get("customer_name") or ""),
            "quantity": r["quantity"],
            "unit_price": r["unit_price"],
            "discount": r["discount"],
            "total_amount": r["total_amount"],
            "channel": r.get("channel"),
            "region": r.get("region"),
            "row_hash": r["row_hash"],
            "source_id": source_id,
            "etl_job_id": etl_job_id,
            "org_id": org_id,
            "ingested_at": ingested_at,
        }
        for r in records
    ]
    loaded = 0
    for batch in _batches(rows):
        stmt = (
            pg_insert(SalesTransaction)
            .values(batch)
            .on_conflict_do_nothing(index_elements=["row_hash"])
            .returning(SalesTransaction.id)
        )
        loaded += len((await db.execute(stmt)).scalars().all())
    return LoadResult(loaded=loaded, skipped_duplicates=len(rows) - loaded)


async def load_expenses(
    db: AsyncSession,
    records: list[dict[str, Any]],
    source_id: uuid.UUID | None,
    etl_job_id: uuid.UUID | None,
    org_id: uuid.UUID | None = None,
) -> LoadResult:
    if not records:
        return LoadResult()
    ingested_at = business_now().replace(tzinfo=None)
    rows = [
        {
            **{
                k: r[k] for k in ("expense_date", "category", "amount", "department", "description")
            },
            "row_hash": r["row_hash"],
            "source_id": source_id,
            "etl_job_id": etl_job_id,
            "org_id": org_id,
            "ingested_at": ingested_at,
        }
        for r in records
    ]
    loaded = 0
    for batch in _batches(rows):
        stmt = (
            pg_insert(Expense)
            .values(batch)
            .on_conflict_do_nothing(index_elements=["row_hash"])
            .returning(Expense.id)
        )
        loaded += len((await db.execute(stmt)).scalars().all())
    return LoadResult(loaded=loaded, skipped_duplicates=len(rows) - loaded)


async def load_inventory(
    db: AsyncSession,
    records: list[dict[str, Any]],
    source_id: uuid.UUID | None,
    etl_job_id: uuid.UUID | None,
    org_id: uuid.UUID | None = None,
) -> LoadResult:
    if not records:
        return LoadResult()
    products = await _ensure_products(db, records, org_id=org_id)
    ingested_at = business_now().replace(tzinfo=None)
    rows = [
        {
            "snapshot_date": r["snapshot_date"],
            "product_id": products[r["sku"]],
            "quantity_on_hand": r["quantity_on_hand"],
            "reorder_level": r["reorder_level"],
            "warehouse": r["warehouse"],
            "source_id": source_id,
            "org_id": org_id,
            "ingested_at": ingested_at,
        }
        for r in records
    ]
    loaded = 0
    for batch in _batches(rows):
        insert_stmt = pg_insert(InventoryLevel).values(batch)
        stmt = insert_stmt.on_conflict_do_update(
            constraint="uq_inventory_snapshot",
            set_={
                "quantity_on_hand": insert_stmt.excluded.quantity_on_hand,
                "reorder_level": insert_stmt.excluded.reorder_level,
                # a re-uploaded snapshot is a fresh ingestion
                "ingested_at": insert_stmt.excluded.ingested_at,
            },
        ).returning(InventoryLevel.id)
        loaded += len((await db.execute(stmt)).scalars().all())
    return LoadResult(loaded=loaded, skipped_duplicates=len(rows) - loaded)


LOADERS = {"sales": load_sales, "finance": load_expenses, "inventory": load_inventory}
