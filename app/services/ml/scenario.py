"""Probabilistic forecasting: Monte Carlo simulation over a fitted model.

Given a target series we fit the best candidate (or ensemble), then bootstrap
residuals to draw many plausible future paths. From those paths we surface
percentile bands (p10/p50/p90) and three scenario bands (optimistic /
base / pessimistic) — the kind of distribution-aware view Power BI's
forecast cards and "what-if" panes provide, but grounded in real residuals.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from app.services.ml import forecasting as fc

logger = logging.getLogger(__name__)


def _fit_point_and_residuals(frame: pd.DataFrame, horizon: int, model: str | None):
    """Return (future_ds, point_forecast, residual_std, chosen_model)."""
    train = frame.iloc[: -fc.HOLDOUT_DAYS] if len(frame) > fc.HOLDOUT_DAYS else frame
    if model in fc.CANDIDATE_NAMES:
        chosen = model
    else:
        chosen, _ = fc.best_candidate(frame)
    forecaster = fc.make_forecaster(chosen)
    forecaster.fit(train)
    future_ds = pd.Series(pd.date_range(frame["ds"].iloc[-1] + pd.Timedelta(days=1), periods=horizon))
    preds = forecaster.predict(future_ds)
    fitted = forecaster.predict(train["ds"].reset_index(drop=True))["yhat"].to_numpy()
    resid = train["y"].to_numpy() - fitted
    resid_std = float(np.std(resid)) or float(np.std(train["y"])) or 1.0
    return future_ds.to_numpy(), preds["yhat"].to_numpy(), resid_std, chosen


def monte_carlo(
    frame: pd.DataFrame,
    horizon: int = 30,
    n_paths: int = 500,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    model: str | None = None,
) -> dict[str, Any]:
    """Simulate ``n_paths`` future trajectories and summarise the distribution.

    Each step's shock is drawn from the fitted residual distribution (with a
    small mean-reversion so paths don't wander unboundedly), then clipped at 0.
    """
    future_ds, point, resid_std, chosen = _fit_point_and_residuals(frame, horizon, model)
    rng = np.random.default_rng(42)
    paths = np.zeros((n_paths, horizon))
    current = np.full(n_paths, float(point[0]))
    for t in range(horizon):
        shock = rng.normal(0, resid_std, size=n_paths)
        current = point[t] + (current - point[t]) * 0.6 + shock
        current = np.clip(current, 0, None)
        paths[:, t] = current

    q = np.quantile(paths, list(quantiles), axis=0)
    p10 = q[0]
    p50 = q[len(quantiles) // 2]
    p90 = q[-1]
    scen = np.quantile(paths, [0.2, 0.5, 0.8], axis=0)
    return {
        "model": chosen,
        "horizon": horizon,
        "n_paths": n_paths,
        "residual_std": round(resid_std, 2),
        "dates": [str(pd.Timestamp(d).date()) for d in future_ds],
        "point": [round(float(v), 2) for v in point],
        "p10": [round(float(v), 2) for v in p10],
        "p50": [round(float(v), 2) for v in p50],
        "p90": [round(float(v), 2) for v in p90],
        "scenarios": {
            "pessimistic": [round(float(v), 2) for v in scen[0]],
            "base": [round(float(v), 2) for v in scen[1]],
            "optimistic": [round(float(v), 2) for v in scen[2]],
        },
        "final": {
            "point": round(float(point[-1]), 2),
            "p10": round(float(p10[-1]), 2),
            "p50": round(float(p50[-1]), 2),
            "p90": round(float(p90[-1]), 2),
            "expected_total": round(float(paths.sum(axis=1).mean()), 2),
            "p10_total": round(float(np.quantile(paths.sum(axis=1), 0.1)), 2),
            "p90_total": round(float(np.quantile(paths.sum(axis=1), 0.9)), 2),
        },
    }
