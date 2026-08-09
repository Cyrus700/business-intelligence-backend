"""Feature preparation: warehouse → tidy daily series for the ML models."""

import json
from datetime import date
from typing import Any

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

# target -> underlying kpi_snapshots metric name
TARGET_METRIC = {
    "revenue_daily": "revenue",
    "orders_daily": "orders",
    "expenses_daily": "expense_total",
}

_SERIES_SQL = (
    "SELECT snapshot_date AS ds, value AS y FROM kpi_snapshots "
    "WHERE metric = :metric AND dimensions = :dim::jsonb ORDER BY snapshot_date"
)

# which segment key(s) each target can be sliced by — mirrors the segment
# columns kpi_builder.py actually populates dimensions with.
SEGMENT_KEYS = {
    "revenue_daily": ("region", "channel"),
    "orders_daily": (),
    "expenses_daily": ("category",),
}

_MARKETING_SQL = (
    "SELECT expense_date AS ds, SUM(amount) AS spend FROM expenses "
    "WHERE category = 'marketing' GROUP BY expense_date ORDER BY expense_date"
)


async def discover_segments(db: AsyncSession, target: str) -> list[dict[str, str]]:
    """Distinct non-empty segment values already present for this target."""
    keys = SEGMENT_KEYS.get(target, ())
    if not keys:
        return []
    metric = TARGET_METRIC[target]
    segments: list[dict[str, str]] = []
    for key in keys:
        rows = (
            await db.execute(
                text(
                    "SELECT DISTINCT dimensions->>:key AS v FROM kpi_snapshots "
                    "WHERE metric = :metric AND dimensions ? :key"
                ),
                {"key": key, "metric": metric},
            )
        ).all()
        segments.extend({key: v} for v, in rows if v)
    return segments


def festival_flags(dates: pd.Series) -> pd.Series:
    """1.0 inside a festival window (incl. 10-day pre-festival ramp), else 0.0."""
    flags = pd.Series(0.0, index=dates.index)
    for start, end in FESTIVAL_WINDOWS:
        s = pd.Timestamp(start) - pd.Timedelta(days=10)
        flags = flags.mask((dates >= s) & (dates <= pd.Timestamp(end)), 1.0)
    return flags


async def _marketing_series(db: AsyncSession, dates: pd.Series) -> pd.Series:
    """Daily marketing expense, reindexed onto `dates` (0 where none logged)."""
    rows = (await db.execute(text(_MARKETING_SQL))).all()
    spend = pd.DataFrame(rows, columns=["ds", "spend"])
    if spend.empty:
        return pd.Series(0.0, index=dates.index)
    spend["ds"] = pd.to_datetime(spend["ds"])
    spend["spend"] = spend["spend"].astype(float)
    merged = pd.DataFrame({"ds": dates}).merge(spend, on="ds", how="left").fillna({"spend": 0.0})
    return merged["spend"]


async def load_series(
    db: AsyncSession, target: str, dimensions: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Continuous daily frame [ds, y, festival, marketing] with missing days as
    explicit zeros.

    Zeros (not interpolation) because a day without transactions genuinely had
    zero revenue — documented modelling choice (docs/05-ml-plan.md).

    `dimensions` slices to one segment (e.g. {"region": "Kathmandu"}); omit
    (or {}) for the whole-business series.
    """
    dims = dimensions or {}
    rows = (
        await db.execute(
            text(_SERIES_SQL), {"metric": TARGET_METRIC[target], "dim": json.dumps(dims)}
        )
    ).all()
    frame = pd.DataFrame(rows, columns=["ds", "y"])
    if frame.empty:
        return frame
    frame["ds"] = pd.to_datetime(frame["ds"])
    frame["y"] = frame["y"].astype(float)
    full = pd.DataFrame({"ds": pd.date_range(frame["ds"].min(), frame["ds"].max(), freq="D")})
    frame = full.merge(frame, on="ds", how="left").fillna({"y": 0.0})
    frame["festival"] = festival_flags(frame["ds"])
    frame["marketing"] = await _marketing_series(db, frame["ds"])
    return frame


def latest_date(frame: pd.DataFrame) -> date:
    return frame["ds"].max().date()
