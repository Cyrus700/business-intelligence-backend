"""Seed the warehouse with the FULL 4-year dataset (see seeds/generate_full_data.py)
through the real ETL pipeline, so seeding exercises the same code path as
production loads.

Usage:
    uv run python seeds/generate_full_data.py   # produce full_*.csv
    uv run python scripts/seed_full_db.py
"""

import asyncio
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.database import get_session_factory  # noqa: E402
from app.models import DataSource  # noqa: E402
from app.services.etl.pipeline import run_frame_pipeline  # noqa: E402

OUTPUT = Path(__file__).resolve().parent.parent / "seeds" / "output"

SOURCES = [
    ("Full sales CSV (4y)", "csv_upload", "sales", "full_sales.csv"),
    ("Full expenses CSV (4y)", "csv_upload", "finance", "full_expenses.csv"),
    ("Full inventory CSV (4y)", "csv_upload", "inventory", "full_inventory.csv"),
]


async def main() -> None:
    async with get_session_factory()() as db:
        for name, kind, domain, file_name in SOURCES:
            source = (await db.execute(select(DataSource).where(DataSource.name == name))).scalar_one_or_none()
            if source is None:
                source = DataSource(name=name, kind=kind, target_domain=domain)
                db.add(source)
                await db.flush()

            frame = pd.read_csv(OUTPUT / file_name)
            result = await run_frame_pipeline(db, domain, frame, trigger="manual", source_id=source.id)
            print(
                f"{file_name:>18}: {result.rows_loaded:,} loaded, "
                f"{result.skipped_duplicates:,} duplicates skipped, "
                f"{result.rows_rejected} rejected"
            )


if __name__ == "__main__":
    asyncio.run(main())
