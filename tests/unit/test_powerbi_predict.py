"""Unit tests for the enhanced prediction engine (no DB required)."""

import numpy as np
import pandas as pd
import pytest

from app.services.ml import forecasting as fc
from app.services.ml.scenario import monte_carlo


def _synthetic(n: int = 220, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2025-01-01", periods=n, freq="D")
    t = np.arange(n)
    y = 1000 + 5 * t + 200 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 30, n)
    return pd.DataFrame({"ds": ds, "y": y})


def test_metrics_known():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 180.0, 330.0])
    m = fc.metrics(y_true, y_pred)
    assert m["mape"] == pytest.approx(10.0, abs=0.5)
    assert m["mae"] == pytest.approx(20.0, abs=1e-6)


def test_naive_seasonal_shape():
    frame = _synthetic()
    f = fc.NaiveSeasonal()
    f.fit(frame.iloc[:-30])
    preds = f.predict(frame["ds"].iloc[-30:].reset_index(drop=True))
    assert len(preds) == 30
    assert "yhat" in preds


def test_ets_and_theta_run():
    frame = _synthetic()
    for name in ("ets", "theta"):
        forecaster = fc.make_forecaster(name)
        forecaster.fit(frame.iloc[:-30])
        preds = forecaster.predict(frame["ds"].iloc[-30:].reset_index(drop=True))
        assert len(preds) == 30
        assert float(preds["yhat"].iloc[0]) >= 0


def test_best_candidate_returns_known_model():
    frame = _synthetic()
    name, mape = fc.best_candidate(frame)
    assert name in fc.CANDIDATE_NAMES
    assert 0 <= mape < 100


def test_monte_carlo_bands_ordered():
    frame = _synthetic()
    out = monte_carlo(frame, horizon=21, n_paths=300)
    assert out["horizon"] == 21
    assert len(out["dates"]) == 21
    assert len(out["p10"]) == 21
    # percentile ordering
    for a, b, c in zip(out["p10"], out["p50"], out["p90"]):
        assert a <= b <= c
    # scenario totals exist and ordered
    assert out["final"]["p10_total"] <= out["final"]["expected_total"] <= out["final"]["p90_total"]


def test_ensemble_forecast_shape():
    frame = _synthetic()
    ens = fc.ensemble_forecast(frame, horizon=14)
    assert len(ens) == 14
    assert "yhat" in ens
