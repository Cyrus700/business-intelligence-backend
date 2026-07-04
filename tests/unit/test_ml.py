"""Deterministic ML unit tests (docs/06-testing-strategy.md Phase 4 obligations).

Prophet is excluded here (slow, stan-backed); it is covered by
scripts/evaluate_ml.py and the integration path. These tests pin the maths that
must never silently change: metrics, the naive baseline, anomaly flagging.
"""

import numpy as np
import pandas as pd

from app.services.ml.anomaly import detect_frame
from app.services.ml.forecasting import NaiveSeasonal, metrics


def make_series(days: int = 400, base: float = 1000.0) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    ds = pd.date_range("2025-01-01", periods=days, freq="D")
    weekday_mult = np.array([1.0, 0.95, 0.95, 1.0, 1.15, 1.3, 0.7])
    y = base * weekday_mult[ds.dayofweek] * (1 + 0.001 * np.arange(days))
    y = y + rng.normal(0, base * 0.03, days)
    return pd.DataFrame({"ds": ds, "y": y, "festival": 0.0})


def test_metrics_computation():
    m = metrics(np.array([100.0, 200.0]), np.array([110.0, 180.0]))
    assert m["mape"] == 10.0  # (10% + 10%) / 2
    assert m["mae"] == 15.0


def test_naive_seasonal_repeats_weekly_pattern():
    frame = make_series()
    model = NaiveSeasonal()
    model.fit(frame)
    future = pd.Series(pd.date_range(frame["ds"].max() + pd.Timedelta(days=1), periods=14))
    preds = model.predict(future)
    # same weekday one week apart should predict (nearly) the same value
    assert abs(preds.iloc[0]["yhat"] - preds.iloc[7]["yhat"]) < 1e-6
    # a Saturday (weekday 5) forecast must exceed a Sunday (weekday 6) one
    by_dow = {ds.dayofweek: yhat for ds, yhat in zip(preds["ds"], preds["yhat"], strict=True)}
    assert by_dow[5] > by_dow[6]


def test_anomaly_detector_catches_injected_spike_and_drop():
    frame = make_series()
    spike_idx, drop_idx = 300, 350
    frame.loc[spike_idx, "y"] *= 3.0
    frame.loc[drop_idx, "y"] *= 0.15
    flagged = detect_frame(frame, "weekly")
    flagged_dates = set(flagged["ds"])
    assert frame.loc[spike_idx, "ds"] in flagged_dates
    assert frame.loc[drop_idx, "ds"] in flagged_dates
    # ...and not the whole series: tolerate at most a couple of noise flags
    assert len(flagged) <= 4


def test_anomaly_detector_ignores_festival_surge():
    frame = make_series()
    # a legitimate festival surge (matches a window in features.FESTIVAL_WINDOWS)
    window = (frame["ds"] >= "2025-09-22") & (frame["ds"] <= "2025-10-02")
    frame.loc[window, "y"] *= 2.1
    frame.loc[window, "festival"] = 1.0
    flagged = detect_frame(frame, "weekly")
    assert not set(frame.loc[window, "ds"]) & set(flagged["ds"])
