"""Feature preparation: warehouse → tidy daily series for the ML models."""

from datetime import date

import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Nepali festival demand windows (mirrors seeds/generate_demo_data.py; the
# generator and the models must agree on the calendar).
FESTIVAL_WINDOWS = [
    ("2023-10-15", "2023-10-24"),
    ("2023-11-10", "2023-11-15"),
    ("2024-10-03", "2024-10-12"),
    ("2024-10-29", "2024-11-03"),
    ("2025-09-22", "2025-10-02"),
    ("2025-10-18", "2025-10-23"),
    ("2026-10-11", "2026-10-20"),
    ("2026-11-06", "2026-11-11"),
    ("2027-09-30", "2027-10-09"),
    ("2027-10-26", "2027-10-31"),
]

SERIES_SQL = {
    "revenue_daily": (
        "SELECT snapshot_date AS ds, value AS y FROM kpi_snapshots "
        "WHERE metric = 'revenue' AND dimensions = '{}'::jsonb ORDER BY snapshot_date"
    ),
    "orders_daily": (
        "SELECT snapshot_date AS ds, value AS y FROM kpi_snapshots "
        "WHERE metric = 'orders' AND dimensions = '{}'::jsonb ORDER BY snapshot_date"
    ),
    "expenses_daily": (
        "SELECT snapshot_date AS ds, value AS y FROM kpi_snapshots "
        "WHERE metric = 'expense_total' AND dimensions = '{}'::jsonb ORDER BY snapshot_date"
    ),
}


def festival_flags(dates: pd.Series) -> pd.Series:
    """1.0 inside a festival window (incl. 10-day pre-festival ramp), else 0.0."""
    flags = pd.Series(0.0, index=dates.index)
    for start, end in FESTIVAL_WINDOWS:
        s = pd.Timestamp(start) - pd.Timedelta(days=10)
        flags = flags.mask((dates >= s) & (dates <= pd.Timestamp(end)), 1.0)
    return flags


async def load_series(db: AsyncSession, target: str) -> pd.DataFrame:
    """Continuous daily frame [ds, y] with missing days as explicit zeros.

    Zeros (not interpolation) because a day without transactions genuinely had
    zero revenue — documented modelling choice (docs/05-ml-plan.md).
    """
    rows = (await db.execute(text(SERIES_SQL[target]))).all()
    frame = pd.DataFrame(rows, columns=["ds", "y"])
    if frame.empty:
        return frame
    frame["ds"] = pd.to_datetime(frame["ds"])
    frame["y"] = frame["y"].astype(float)
    full = pd.DataFrame({"ds": pd.date_range(frame["ds"].min(), frame["ds"].max(), freq="D")})
    frame = full.merge(frame, on="ds", how="left").fillna({"y": 0.0})
    frame["festival"] = festival_flags(frame["ds"])
    return frame


def latest_date(frame: pd.DataFrame) -> date:
    return frame["ds"].max().date()
