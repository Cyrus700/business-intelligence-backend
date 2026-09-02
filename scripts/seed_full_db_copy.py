"""Fastest bulk seeder for the FULL 4-year dataset, using PostgreSQL COPY.

The remote database has high per-statement latency, so the row-by-row / 2k-batch
ETL loader cannot finish 355k inserts in reasonable time. This seeder still
reuses the *real* transform code (canonical records + row_hashes identical to
production), then streams the rows in via the asyncpg COPY protocol — a single
command for the whole table, no per-row round trips.

Idempotent: for each load it first deletes rows belonging to its DataSource, so
re-running cleanly re-seeds. Products & customers are upserted by natural key.

Usage:
    uv run python seeds/generate_full_data.py
    uv run python scripts/seed_full_db_copy.py
"""

import asyncio
import re
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import asyncpg
import pandas as pd
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.core.database import get_session_factory  # noqa: E402
from app.models import Customer, DataSource, Product  # noqa: E402
from app.services.etl.domains import transform_frame  # noqa: E402

OUTPUT = Path(__file__).resolve().parent.parent / "seeds" / "output"

SOURCES = [
    ("Full sales CSV (4y)", "csv_upload", "sales", "full_sales.csv"),
    ("Full expenses CSV (4y)", "csv_upload", "finance", "full_expenses.csv"),
    ("Full inventory CSV (4y)", "csv_upload", "inventory", "full_inventory.csv"),
]


def _f(v):
    """Coerce a Decimal/money value to float for COPY (Numeric(14,2) safe)."""
    if isinstance(v, Decimal):
        return float(v)
    return v


def _dedupe(records):
    """Drop rows sharing a row_hash (mirrors the loader's ON CONFLICT DO NOTHING)."""
    seen = set()
    out = []
    for r in records:
        h = r["row_hash"]
        if h in seen:
            continue
        seen.add(h)
        out.append(r)
    return out


async def _get_or_create_source(db, name, kind, domain):
    src = (await db.execute(select(DataSource).where(DataSource.name == name))).scalar_one_or_none()
    if src is None:
        src = DataSource(name=name, kind=kind, target_domain=domain)
        db.add(src)
        await db.flush()
    return src


async def _product_map(db):
    rows = (await db.execute(select(Product.sku, Product.id))).all()
    return {sku: pid for sku, pid in rows}


async def _customer_map(db):
    rows = (await db.execute(select(Customer.name, Customer.id))).all()
    return {name: cid for name, cid in rows}


async def _upsert_products(db, records):
    existing = await _product_map(db)
    wanted = {}
    for r in records:
        wanted.setdefault(r["sku"], r)
    new = [
        Product(
            sku=r["sku"],
            name=r.get("product_name") or r["sku"],
            category=r.get("category"),
            unit_price=_f(r.get("unit_price")),
        )
        for sku, r in wanted.items()
        if sku not in existing
    ]
    if new:
        db.add_all(new)
        await db.flush()
    return await _product_map(db)


async def _upsert_customers(db, records):
    existing = await _customer_map(db)
    wanted = {}
    for r in records:
        nm = r.get("customer_name")
        if nm:
            wanted.setdefault(nm, r)
    new = [
        Customer(name=nm, segment=r.get("segment"), city=r.get("city"), region=r.get("region"))
        for nm, r in wanted.items()
        if nm not in existing
    ]
    if new:
        db.add_all(new)
        await db.flush()
    return await _customer_map(db)


async def main() -> None:
    settings = get_settings()
    # postgresql+asyncpg://...  ->  postgresql://...  for the raw asyncpg driver
    pg_dsn = re.sub(r"^\w+(\+\w+)?://", "postgresql://", settings.database_url)
    pg_conn = await asyncpg.connect(pg_dsn)

    async with get_session_factory()() as db:
        ingested = datetime.now(UTC).replace(tzinfo=None)

        # ---- SALES ----
        name, kind, domain, file_name = SOURCES[0]
        src = await _get_or_create_source(db, name, kind, domain)
        frame = pd.read_csv(OUTPUT / file_name)
        res = transform_frame(domain, frame)
        res.records = _dedupe(res.records)
        pmap = await _upsert_products(db, res.records)
        cmap = await _upsert_customers(db, res.records)
        # Commit so the separate asyncpg COPY connection can see these rows.
        await db.commit()
        await pg_conn.execute("DELETE FROM sales_transactions WHERE source_id = $1", src.id)
        rows = [
            (
                r["txn_date"],
                pmap.get(r["sku"]),
                cmap.get(r.get("customer_name") or ""),
                int(r["quantity"]),
                _f(r["unit_price"]),
                _f(r["discount"]),
                _f(r["total_amount"]),
                r.get("channel"),
                r.get("region"),
                r["row_hash"],
                src.id,
                ingested,
            )
            for r in res.records
        ]
        await pg_conn.copy_records_to_table(
            "sales_transactions",
            records=rows,
            columns=[
                "txn_date",
                "product_id",
                "customer_id",
                "quantity",
                "unit_price",
                "discount",
                "total_amount",
                "channel",
                "region",
                "row_hash",
                "source_id",
                "ingested_at",
            ],
        )
        print(f"full_sales.csv   : {len(rows):>9,} COPY-loaded")

        # ---- EXPENSES ----
        name, kind, domain, file_name = SOURCES[1]
        src = await _get_or_create_source(db, name, kind, domain)
        await db.commit()
        frame = pd.read_csv(OUTPUT / file_name)
        res = transform_frame(domain, frame)
        res.records = _dedupe(res.records)
        await pg_conn.execute("DELETE FROM expenses WHERE source_id = $1", src.id)
        rows = [
            (
                r["expense_date"],
                r["category"],
                _f(r["amount"]),
                r.get("department"),
                r.get("description"),
                r["row_hash"],
                src.id,
                ingested,
            )
            for r in res.records
        ]
        await pg_conn.copy_records_to_table(
            "expenses",
            records=rows,
            columns=[
                "expense_date",
                "category",
                "amount",
                "department",
                "description",
                "row_hash",
                "source_id",
                "ingested_at",
            ],
        )
        print(f"full_expenses.csv : {len(rows):>9,} COPY-loaded")

        # ---- INVENTORY ----
        name, kind, domain, file_name = SOURCES[2]
        src = await _get_or_create_source(db, name, kind, domain)
        await db.commit()
        frame = pd.read_csv(OUTPUT / file_name)
        res = transform_frame(domain, frame)
        pmap = await _product_map(db)
        await pg_conn.execute("DELETE FROM inventory_levels WHERE source_id = $1", src.id)
        rows = [
            (
                r["snapshot_date"],
                pmap[r["sku"]],
                int(r["quantity_on_hand"]),
                int(r["reorder_level"]),
                r.get("warehouse") or "main",
                src.id,
                ingested,
            )
            for r in res.records
        ]
        await pg_conn.copy_records_to_table(
            "inventory_levels",
            records=rows,
            columns=[
                "snapshot_date",
                "product_id",
                "quantity_on_hand",
                "reorder_level",
                "warehouse",
                "source_id",
                "ingested_at",
            ],
        )
        print(f"full_inventory.csv: {len(rows):>9,} COPY-loaded")

        # ---- Rebuild KPIs + refresh derived layer once ----
        from datetime import date as _d

        from app.services.analytics.kpi_builder import rebuild_kpi_snapshots
        from app.services.etl.refresh import refresh_derived

        span_lo = _d.fromisoformat(frame["date"].min())
        span_hi = _d.fromisoformat(frame["date"].max())
        print("Rebuilding KPI snapshots ...")
        await rebuild_kpi_snapshots(db, span_lo, span_hi)
        await db.commit()
        print("Refreshing derived layer (anomalies / insights / alerts) ...")
        refresh = await refresh_derived(db, span_lo, span_hi)
        print("Refresh:", refresh.as_log())

    await pg_conn.close()
    print("Full 4-year dataset seeded via COPY.")


if __name__ == "__main__":
    asyncio.run(main())
