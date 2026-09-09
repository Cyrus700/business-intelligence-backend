"""Trend analysis: direction + strength from a rolling linear fit."""

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.ml.features import load_series

TARGETS = {"revenue": "revenue_daily", "orders": "orders_daily", "expenses": "expenses_daily"}

try:
    from scipy import stats as scipy_stats

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


async def trend_summary(db: AsyncSession, metric: str, window_days: int = 90, org_id=None) -> dict | None:
    frame = await load_series(db, TARGETS[metric], org_id=org_id)
    if frame.empty:
        return None
    # allow partial window: need at least 14 days, but prefer window_days
    n_available = len(frame)
    if n_available < 14:
        return None
    # use min(window_days, n_available) — partial window
    effective_window = min(window_days, n_available)
    tail = frame.tail(effective_window)
    n = len(tail)
    if n < 7:
        return None
    x = np.arange(n, dtype=float)
    y = tail["y"].to_numpy(dtype=float)

    # Use scipy linregress for p-value, confidence interval, stderr
    if HAS_SCIPY:
        try:
            res = scipy_stats.linregress(x, y)
            slope = float(res.slope)
            intercept = float(res.intercept)
            r_value = float(res.rvalue) if np.isfinite(res.rvalue) else 0.0
            p_value = float(res.pvalue) if np.isfinite(res.pvalue) else 1.0
            stderr = float(res.stderr) if res.stderr is not None and np.isfinite(res.stderr) else 0.0
            # confidence interval 95% for slope: slope ± t*stderr, t ~ 1.96 for large n, else t distribution
            if stderr and n > 2:
                # use t critical for 95% CI
                try:
                    t_val = float(scipy_stats.t.ppf(0.975, df=n - 2))
                except Exception:
                    t_val = 1.96
                ci_low = slope - t_val * stderr
                ci_high = slope + t_val * stderr
            else:
                ci_low = ci_high = slope
        except Exception:
            # fallback to polyfit
            slope, intercept = np.polyfit(x, y, 1)
            slope = float(slope)
            intercept = float(intercept)
            r = np.corrcoef(x, y)[0, 1] if y.std() > 0 and x.std() > 0 else 0.0
            r_value = float(r) if np.isfinite(r) else 0.0
            p_value = 1.0
            stderr = 0.0
            ci_low = ci_high = slope
    else:
        slope, intercept = np.polyfit(x, y, 1)
        slope = float(slope)
        intercept = float(intercept)
        r = np.corrcoef(x, y)[0, 1] if y.std() > 0 and x.std() > 0 else 0.0
        r_value = float(r) if np.isfinite(r) else 0.0
        p_value = 1.0
        stderr = 0.0
        ci_low = ci_high = slope

    mean = float(y.mean()) if len(y) else 0.0
    # handle mean < 1e-3 reporting absolute delta instead of pct
    if abs(mean) < 1e-3:
        weekly_abs = slope * 7
        weekly_pct = 0.0
        # direction based on absolute slope
        # Use absolute change threshold ~ 1 unit per week?
        direction = "rising" if weekly_abs > 1e-6 else "falling" if weekly_abs < -1e-6 else "flat"
        use_abs = True
    else:
        weekly_pct = slope * 7 / mean * 100  # % of mean per week
        weekly_abs = slope * 7
        direction = "rising" if weekly_pct > 1 else "falling" if weekly_pct < -1 else "flat"
        use_abs = False

    # strength from correlation or p-value
    r_abs = abs(r_value) if np.isfinite(r_value) else 0.0
    # also consider p-value for significance
    strength_r = round(float(r_abs), 2)

    # last ds for front-end
    try:
        last_ds = str(tail["ds"].iloc[-1].date())
    except Exception:
        last_ds = None

    # points for sparkline: last 30 or full tail?
    points = []
    try:
        for idx, row in tail.reset_index(drop=True).iterrows():
            points.append({"ds": str(row["ds"].date()) if hasattr(row["ds"], "date") else str(row["ds"]), "y": round(float(row["y"]), 2)})
    except Exception:
        points = []

    out: dict = {
        "metric": metric,
        "window_days": window_days,
        "effective_window": effective_window,
        "direction": direction,
        "weekly_change_pct": round(float(weekly_pct), 2),
        "weekly_change_abs": round(float(weekly_abs), 2),
        "strength_r": strength_r,
        "current_level": round(float(y[-7:].mean()), 2),
        "slope": round(float(slope), 4),
        "intercept": round(float(intercept), 2),
        "ci_low": round(float(ci_low), 4),
        "ci_high": round(float(ci_high), 4),
        "p_value": round(float(p_value), 4) if np.isfinite(p_value) else 1.0,
        "stderr": round(float(stderr), 4) if np.isfinite(stderr) else 0.0,
        "r_value": round(float(r_value), 3),
        "n": int(n),
        "last_ds": last_ds,
        "points": points,
        "mean": round(float(mean), 2),
        "use_abs": use_abs,
    }
    # backward compat aliases
    if use_abs:
        # also expose weekly_change_pct as abs for compatibility
        pass
    return out
