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

# Minimum history for a stable residual distribution. Below this the holdout
# estimate is too noisy and bands will be misleading.
MIN_HISTORY = 180


def _fit_point_and_residuals(
    frame: pd.DataFrame,
    horizon: int,
    model: str | None,
    seed: int | None = None,
):
    """Return (future_ds, point_forecast, resid, resid_std, chosen_model, mape).

    * length guard >= MIN_HISTORY (180)
    * holdout residuals (not in-sample) — fit on train, score on test
    * ddof=1 for unbiased std
    * exposes mape of chosen model for band metadata
    """
    if len(frame) < MIN_HISTORY:
        raise ValueError(f"need at least {MIN_HISTORY} days of history (got {len(frame)})")
    # ensure sorted by ds
    frame = frame.sort_values("ds").reset_index(drop=True)
    train = frame.iloc[:-fc.HOLDOUT_DAYS]
    test = frame.iloc[-fc.HOLDOUT_DAYS:]
    if len(train) < 30 or len(test) < 7:
        raise ValueError("insufficient train/test split for holdout residuals")
    if model in fc.CANDIDATE_NAMES:
        chosen = model
        # compute mape for metadata even when model is forced
        try:
            evals = fc.evaluate_candidates(frame)
            by_name = {e.model_name: e for e in evals}
            mape = by_name.get(chosen, None)
            mape_val = float(mape.metrics["mape"]) if mape and not np.isnan(mape.metrics["mape"]) else None
        except Exception:
            mape_val = None
    else:
        chosen, mape_val = fc.best_candidate(frame)
        if mape_val is not None and np.isnan(mape_val):
            mape_val = None
    forecaster = fc.make_forecaster(chosen)
    forecaster.fit(train)
    future_ds = pd.Series(pd.date_range(frame["ds"].iloc[-1] + pd.Timedelta(days=1), periods=horizon))
    preds = forecaster.predict(future_ds)
    point = preds["yhat"].to_numpy(dtype=float)

    # holdout residuals: predict test ds with train-fitted model
    test_preds = forecaster.predict(test["ds"].reset_index(drop=True))
    resid = test["y"].to_numpy(dtype=float) - test_preds["yhat"].to_numpy(dtype=float)
    # drop NaNs (e.g. failed preds)
    resid = resid[~np.isnan(resid)]
    if len(resid) == 0:
        # fallback to in-sample if holdout all NaN — still better than 1.0
        fitted = forecaster.predict(train["ds"].reset_index(drop=True))["yhat"].to_numpy(dtype=float)
        resid = train["y"].to_numpy(dtype=float) - fitted
        resid = resid[~np.isnan(resid)]
    if len(resid) == 0:
        resid = np.array([0.0])
    # unbiased std
    if len(resid) > 1:
        resid_std = float(np.nanstd(resid, ddof=1))
    else:
        resid_std = float(np.abs(resid[0])) if len(resid) else 0.0
    # fallback if resid_std is 0 or nan
    if not np.isfinite(resid_std) or resid_std == 0.0:
        resid_std = float(np.nanstd(train["y"].to_numpy(dtype=float), ddof=1) or 1.0)
        if not np.isfinite(resid_std) or resid_std == 0.0:
            resid_std = 1.0
    return future_ds.to_numpy(), point, resid, resid_std, chosen, mape_val


def monte_carlo(
    frame: pd.DataFrame,
    horizon: int = 30,
    n_paths: int = 500,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    model: str | None = None,
    seed: int | None = None,
    mean_reversion: float = 0.0,
) -> dict[str, Any]:
    """Simulate ``n_paths`` future trajectories and summarise the distribution.

    Each step's shock is drawn empirically from holdout residuals via
    ``rng.choice(resid)`` (bootstrap, not Gaussian), so heavy tails and
    asymmetry are preserved.  ``mean_reversion`` pulls the path toward the
    point forecast (0.0 = independent shocks, 1.0 = full reversion). The
    previous hard-coded 0.6 is removed; callers can estimate it or pass 0.0.

    Clip bias is removed: paths are allowed to wander below 0 during
    simulation; final percentile bands are clipped for display with a warning
    flag so the truncation is visible rather than silently biasing totals.
    """
    future_ds, point, resid, resid_std, chosen, model_mape = _fit_point_and_residuals(
        frame, horizon, model, seed=seed
    )
    rng = np.random.default_rng(seed)
    # empirical bootstrap — preserves distribution shape
    n_resid = len(resid)
    # band metadata helpers
    resid_dist = "empirical_bootstrap"
    # mean of residuals (should be ~0; track for bias)
    resid_mean = float(np.mean(resid)) if len(resid) else 0.0

    paths = np.zeros((n_paths, horizon))
    current = np.full(n_paths, float(point[0]))

    # Pre-sample all shocks empirically for speed and reproducibility
    # shape (n_paths, horizon) of choices from resid
    if n_resid > 0:
        # vectorized choice: indices
        shock_indices = rng.integers(0, n_resid, size=(n_paths, horizon))
        shocks = resid[shock_indices]
    else:
        shocks = rng.normal(0, resid_std, size=(n_paths, horizon))

    for t in range(horizon):
        shock_t = shocks[:, t]
        if t == 0:
            # first step: point[0] + shock (mean_reversion has no prior deviation)
            current = point[t] + shock_t
        else:
            if mean_reversion != 0.0:
                # AR(1)-style reversion: deviation decays by mean_reversion
                deviation = current - point[t - 1]
                current = point[t] + deviation * mean_reversion + shock_t
            else:
                current = point[t] + shock_t
        # NO hard clip inside loop — avoids upward bias. Negatives are allowed;
        # final bands will be clipped for display with metadata.
        paths[:, t] = current

    # --- quantile mapping by value, not position ---
    # always compute canonical p10/p50/p90 at 0.1/0.5/0.9 irrespective of quantiles arg
    p10 = np.quantile(paths, 0.1, axis=0)
    p50 = np.quantile(paths, 0.5, axis=0)
    p90 = np.quantile(paths, 0.9, axis=0)
    # additional quantiles dict for callers that passed custom quantiles
    q_map: dict[float, np.ndarray] = {}
    if quantiles:
        # Use sorted unique for quantile calc, map back by value
        uniq_q = sorted(set(float(q) for q in quantiles))
        if uniq_q:
            q_arr = np.quantile(paths, uniq_q, axis=0)
            # q_arr shape (len(uniq_q), horizon) or (horizon,) if single quantile
            if q_arr.ndim == 1 and len(uniq_q) > 1:
                # defensive
                pass
            for idx, qv in enumerate(uniq_q):
                # q_arr[idx] is array of horizon
                arr = q_arr[idx] if q_arr.ndim == 2 else q_arr
                q_map[qv] = arr
            # ensure p10/p50/p90 use canonical values, not positional fallback
            # if caller included those quantiles, they will match; if not, we keep canonical
    scen_pess = np.quantile(paths, 0.2, axis=0)
    scen_base = np.quantile(paths, 0.5, axis=0)
    scen_opt = np.quantile(paths, 0.8, axis=0)

    # clip for display (revenue cannot be negative) but track bias
    p10_clipped = np.clip(p10, 0, None)
    p50_clipped = np.clip(p50, 0, None)
    p90_clipped = np.clip(p90, 0, None)
    scen_pess_c = np.clip(scen_pess, 0, None)
    scen_base_c = np.clip(scen_base, 0, None)
    scen_opt_c = np.clip(scen_opt, 0, None)
    point_clipped = np.clip(point, 0, None)

    # negatives fraction for warning
    neg_frac = float(np.mean(paths < 0)) if paths.size else 0.0
    # band width (avg spread)
    band_width = float(np.mean(p90_clipped - p10_clipped))
    # point vs clipped divergence indicates clipping bias
    clip_bias = float(np.mean(np.maximum(0, -np.quantile(paths, 0.1, axis=0)))) if neg_frac > 0 else 0.0

    warning = None
    # choose most relevant warning
    if neg_frac > 0.05:
        warning = f"clipped {neg_frac*100:.1f}% negative path values to 0; totals may be upward biased by ~{clip_bias:.2f}"
    elif resid_std > float(np.mean(np.abs(point)) or 1.0) * 0.5:
        warning = "high residual variance vs level — bands are wide, interpret with caution"
    elif model_mape is not None and model_mape > 30:
        warning = f"model MAPE {model_mape:.1f}% is high — scenario bands reflect large forecast error"
    elif n_resid < 30:
        warning = "few holdout residuals — bootstrap distribution may be under-sampled"

    # confidence for p10-p90 band is 80%
    confidence = 0.8

    # path totals distribution (use unclipped paths for totals, then clip totals at 0)
    path_totals = paths.sum(axis=1)
    # totals clipped at 0 for display
    expected_total = float(np.mean(np.clip(path_totals, 0, None)))
    # quantile totals also clipped
    p10_total = float(np.quantile(np.clip(path_totals, 0, None), 0.1))
    p90_total = float(np.quantile(np.clip(path_totals, 0, None), 0.9))

    return {
        "model": chosen,
        "horizon": horizon,
        "n_paths": n_paths,
        "residual_std": round(resid_std, 2),
        "resid_mean": round(resid_mean, 2),
        "dates": [str(pd.Timestamp(d).date()) for d in future_ds],
        "point": [round(float(v), 2) for v in point_clipped],
        "p10": [round(float(v), 2) for v in p10_clipped],
        "p50": [round(float(v), 2) for v in p50_clipped],
        "p90": [round(float(v), 2) for v in p90_clipped],
        "scenarios": {
            "pessimistic": [round(float(v), 2) for v in scen_pess_c],
            "base": [round(float(v), 2) for v in scen_base_c],
            "optimistic": [round(float(v), 2) for v in scen_opt_c],
        },
        "final": {
            "point": round(float(point_clipped[-1]), 2),
            "p10": round(float(p10_clipped[-1]), 2),
            "p50": round(float(p50_clipped[-1]), 2),
            "p90": round(float(p90_clipped[-1]), 2),
            "expected_total": round(expected_total, 2),
            "p10_total": round(p10_total, 2),
            "p90_total": round(p90_total, 2),
        },
        # --- band metadata ---
        "confidence": confidence,
        "resid_dist": resid_dist,
        "model_mape": round(float(model_mape), 2) if model_mape is not None else None,
        "band_width": round(band_width, 2),
        "warning": warning,
        "band_method": "empirical_bootstrap",
        "neg_fraction": round(neg_frac, 4),
        "quantiles": {str(k): [round(float(v), 2) for v in arr] for k, arr in q_map.items()},
    }
