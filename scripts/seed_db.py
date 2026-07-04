"""Seed the warehouse with the generated demo dataset — through the real ETL pipeline,
so seeding exercises the same code path as production loads.

Usage:
    uv run python seeds/generate_demo_data.py   # once, to produce the CSVs
    uv run python scripts/seed_db.py
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
    ("Baseline sales CSV", "csv_upload", "sales", "sales.csv"),
    ("Baseline expenses CSV", "csv_upload", "finance", "expenses.csv"),
    ("Baseline inventory CSV", "csv_upload", "inventory", "inventory.csv"),
]


async def main() -> None:
    async with get_session_factory()() as db:
        for name, kind, domain, file_name in SOURCES:
            source = (
                await db.execute(select(DataSource).where(DataSource.name == name))
            ).scalar_one_or_none()
            if source is None:
                source = DataSource(name=name, kind=kind, target_domain=domain)
                db.add(source)
                await db.flush()

            frame = pd.read_csv(OUTPUT / file_name)
            result = await run_frame_pipeline(
                db, domain, frame, trigger="manual", source_id=source.id
            )
            print(
                f"{file_name:>16}: {result.rows_loaded:,} loaded, "
                f"{result.skipped_duplicates:,} duplicates skipped, "
                f"{result.rows_rejected} rejected"
            )


if __name__ == "__main__":
    asyncio.run(main())
