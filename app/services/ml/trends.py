"""Trend analysis: direction + strength from a rolling linear fit."""

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.ml.features import load_series

TARGETS = {"revenue": "revenue_daily", "orders": "orders_daily", "expenses": "expenses_daily"}


async def trend_summary(db: AsyncSession, metric: str, window_days: int = 90) -> dict | None:
    frame = await load_series(db, TARGETS[metric])
    if len(frame) < window_days:
        return None
    tail = frame.tail(window_days)
    x = np.arange(len(tail), dtype=float)
    y = tail["y"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    mean = y.mean() or 1e-9
    weekly_pct = slope * 7 / mean * 100  # % of mean per week
    direction = "rising" if weekly_pct > 1 else "falling" if weekly_pct < -1 else "flat"
    r = np.corrcoef(x, y)[0, 1] if y.std() > 0 else 0.0
    return {
        "metric": metric,
        "window_days": window_days,
        "direction": direction,
        "weekly_change_pct": round(float(weekly_pct), 2),
        "strength_r": round(float(abs(r)), 2),
        "current_level": round(float(y[-7:].mean()), 2),
    }
