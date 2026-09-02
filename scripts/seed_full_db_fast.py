"""Fast bulk seeder for the FULL 4-year dataset.

Reuses the *real* ETL transform + loader code paths (so row_hashes, products
and customers are created exactly as in production), but loads all three
domains first and then runs the KPI rebuild and the derived-layer refresh a
single time at the end. This avoids the per-pipeline anomaly/insight sweep that
makes a 355k-row load time out on a remote database.

Idempotent: loaders upsert on row_hash / natural key, so re-running safely
skips what is already present.

Usage:
    uv run python seeds/generate_full_data.py   # produce full_*.csv
    uv run python scripts/seed_full_db_fast.py
"""

import asyncio
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.database import get_session_factory  # noqa: E402
from app.models import DataSource  # noqa: E402
from app.services.etl.domains import transform_frame  # noqa: E402
from app.services.etl.loader import load_sales, load_expenses, load_inventory  # noqa: E402
from app.services.analytics.kpi_builder import rebuild_kpi_snapshots  # noqa: E402
from app.services.etl.refresh import refresh_derived  # noqa: E402

OUTPUT = Path(__file__).resolve().parent.parent / "seeds" / "output"

SOURCES = [
    ("Full sales CSV (4y)", "csv_upload", "sales", "full_sales.csv", load_sales),
    ("Full expenses CSV (4y)", "csv_upload", "finance", "full_expenses.csv", load_expenses),
    ("Full inventory CSV (4y)", "csv_upload", "inventory", "full_inventory.csv", load_inventory),
]


async def main() -> None:
    min_date = date.max
    max_date = date.min
    async with get_session_factory()() as db:
        for name, kind, domain, file_name, loader in SOURCES:
            source = (
                await db.execute(select(DataSource).where(DataSource.name == name))
            ).scalar_one_or_none()
            if source is None:
                source = DataSource(name=name, kind=kind, target_domain=domain)
                db.add(source)
                await db.flush()

            frame = pd.read_csv(OUTPUT / file_name)
            result = transform_frame(domain, frame)
            load_result = await loader(db, result.records, source.id, None)
            await db.commit()

            date_field = {"sales": "txn_date", "finance": "expense_date", "inventory": "snapshot_date"}[domain]
            dates = [r[date_field] for r in result.records]
            min_date = min(min_date, min(dates))
            max_date = max(max_date, max(dates))
            print(
                f"{file_name:>18}: {load_result.loaded:,} loaded, "
                f"{load_result.skipped_duplicates:,} skipped, "
                f"{len(result.errors)} rejected"
            )

        print(f"Rebuilding KPI snapshots over {min_date} → {max_date} ...")
        await rebuild_kpi_snapshots(db, min_date, max_date)
        await db.commit()

        print("Refreshing derived layer (anomalies / insights / alerts) once ...")
        refresh = await refresh_derived(db, min_date, max_date)
        print("Refresh:", refresh.as_log())

    print("Full dataset seeded.")


if __name__ == "__main__":
    asyncio.run(main())
