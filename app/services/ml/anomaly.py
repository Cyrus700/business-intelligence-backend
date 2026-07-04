"""Anomaly detection: Isolation Forest + robust z-score cross-check.

Two seasonality modes:
- weekly  (revenue/orders): expected = same-weekday rolling median; STL residuals.
  Festival windows are excluded — festival demand surges are *expected* in the
  Nepali retail calendar, not anomalies.
- monthly (expenses): expected = same day-of-month rolling median, because rent
  (1st), utilities (10th) and salaries (25th) dominate the series.

Both methods vote; agreement → high severity. Every anomaly stores expected vs
observed and the method scores (R5 explainability).
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from statsmodels.tsa.seasonal import STL

from app.models import Anomaly
from app.services.ml.features import load_series

logger = logging.getLogger(__name__)

CONTAMINATION = 0.015  # tuned against seeds/injected_anomalies.json (scripts/evaluate_ml.py)
Z_THRESHOLD = 3.5
MIN_PCT_DEV = 0.4  # ignore statistically-odd but business-trivial wiggles
EXTREME_PCT_DEV = 1.5  # single-method flags need an unambiguous business-scale deviation
# materiality gate: |y - expected| as a fraction of the series mean; monthly
# series (expenses) need a higher bar because ordinary days are small and noisy
MIN_ABS_DEV_FRAC = {"weekly": 0.4, "monthly": 2.0}

TARGET_CONFIG = {
    "revenue_daily": {"metric": "revenue", "season": "weekly"},
    "expenses_daily": {"metric": "expense_total", "season": "monthly"},
}


def _robust_z(residuals: pd.Series) -> pd.Series:
    mad = np.median(np.abs(residuals - np.median(residuals))) or 1e-9
    return 0.6745 * (residuals - np.median(residuals)) / mad


def detect_frame(frame: pd.DataFrame, season: str = "weekly") -> pd.DataFrame:
    """Score a [ds, y, festival] daily frame; returns flagged rows."""
    df = frame.copy().reset_index(drop=True)

    if season == "weekly":
        group = df["ds"].dt.dayofweek
        df["expected"] = df.groupby(group)["y"].transform(
            lambda s: s.rolling(8, min_periods=3).median().shift(1)
        )
        df["z"] = _robust_z(STL(df["y"], period=7, robust=True).fit().resid)
    else:  # monthly
        group = df["ds"].dt.day
        df["expected"] = df.groupby(group)["y"].transform(
            lambda s: s.rolling(4, min_periods=2).median().shift(1)
        )
        df["z"] = _robust_z((df["y"] - df["expected"]).fillna(0.0))

    df["pct_dev"] = (df["y"] - df["expected"]) / df["expected"].replace(0, np.nan)

    feats = pd.DataFrame(
        {
            "y": df["y"],
            "pct_dev": df["pct_dev"].fillna(0),
            "z": df["z"],
            "festival": df.get("festival", pd.Series(0.0, index=df.index)),
        }
    )
    forest = IsolationForest(contamination=CONTAMINATION, random_state=42)
    df["if_score"] = forest.fit(feats).decision_function(feats)

    abs_dev = (df["y"] - df["expected"]).abs()
    material = abs_dev >= MIN_ABS_DEV_FRAC[season] * df["y"].mean()
    big_dev = (df["pct_dev"].abs() >= MIN_PCT_DEV) & material
    df["stl_flag"] = (df["z"].abs() >= Z_THRESHOLD) & big_dev
    df["if_flag"] = (df["if_score"] < 0) & big_dev
    if season == "weekly" and "festival" in df:
        in_festival = df["festival"] > 0
        df.loc[in_festival, ["stl_flag", "if_flag"]] = False

    # final policy (tuned on injected labels): both methods agree, or a single
    # method with an extreme deviation — cuts single-method noise flags
    agree = df["stl_flag"] & df["if_flag"]
    extreme = (df["stl_flag"] | df["if_flag"]) & (df["pct_dev"].abs() >= EXTREME_PCT_DEV)
    flagged = df[agree | extreme].copy()
    flagged["severity"] = np.where(
        flagged["stl_flag"] & flagged["if_flag"],
        "high",
        np.where(flagged["pct_dev"].abs() > 1.0, "medium", "low"),
    )
    return flagged


async def scan_target(db: AsyncSession, target: str, lookback_days: int | None = None) -> int:
    """Detect anomalies for a target and persist new ones (deduped by metric+date)."""
    config = TARGET_CONFIG[target]
    frame = await load_series(db, target)
    if len(frame) < 60:
        return 0
    flagged = detect_frame(frame, config["season"])
    if lookback_days is not None:
        cutoff = frame["ds"].max() - pd.Timedelta(days=lookback_days)
        flagged = flagged[flagged["ds"] >= cutoff]
    if flagged.empty:
        return 0

    metric = config["metric"]
    existing = {
        (a.metric, a.context.get("date"))
        for a in (await db.execute(select(Anomaly).where(Anomaly.metric == metric))).scalars()
        if a.context
    }
    created = 0
    for _, row in flagged.iterrows():
        day = row["ds"].date().isoformat()
        if (metric, day) in existing:
            continue
        direction = "above" if (row["pct_dev"] or 0) >= 0 else "below"
        context: dict[str, Any] = {
            "date": day,
            "direction": direction,
            "pct_deviation": None
            if pd.isna(row["pct_dev"])
            else round(float(row["pct_dev"]) * 100, 1),
            "methods": {
                "robust_zscore": round(float(row["z"]), 2),
                "isolation_forest": round(float(row["if_score"]), 4),
            },
        }
        db.add(
            Anomaly(
                metric=metric,
                observed_value=round(float(row["y"]), 2),
                expected_value=None
                if pd.isna(row["expected"])
                else round(float(row["expected"]), 2),
                deviation_score=round(float(abs(row["z"])), 2),
                severity=row["severity"],
                context=context,
            )
        )
        created += 1
    await db.commit()
    logger.info("%s: %d new anomalies", target, created)
    return created


async def scan_all(db: AsyncSession, lookback_days: int | None = None) -> int:
    total = 0
    for target in TARGET_CONFIG:
        total += await scan_target(db, target, lookback_days)
    return total
