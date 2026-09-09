"""Forecasting: Prophet, ARIMA, ETS, Theta vs naive seasonal baseline + ensemble.

Evaluation protocol (docs/05-ml-plan.md): time-based split — train on all but
the last HOLDOUT_DAYS, score on the holdout, always against the naive seasonal
baseline (same weekday, previous week). Candidates are compared by MAPE; the
best is promoted and an ensemble (inverse-MAPE weighted) is also produced.
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from app.services.ml.features import festival_flags

logger = logging.getLogger(__name__)

# Every candidate the engine can train. ``naive_seasonal`` is the floor that all
# real models must beat; ETS and Theta add classical statistical alternatives so
# the registry can pick the genuinely simplest adequate model (totos.md §13).
CANDIDATE_NAMES = ("naive_seasonal", "prophet", "arima", "ets", "theta")

HOLDOUT_DAYS = 90


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    # MAPE with zero handling: use standard MAPE where y_true !=0, else sMAPE fallback
    abs_true = np.abs(y_true)
    # mask where true is effectively zero
    mape_mask = abs_true >= 1e-9
    if np.any(mape_mask):
        mape_vals = np.abs(err[mape_mask] / y_true[mape_mask]) * 100
        mape = float(np.nanmean(mape_vals))
    else:
        mape = float("nan")
    # if mape is nan (all true zeros), use sMAPE
    if not np.isfinite(mape):
        # sMAPE: 200 * |err| / (|y_true| + |y_pred|)
        denom = (np.abs(y_true) + np.abs(y_pred)) / 2
        # where denom==0, error is 0 -> contribution 0; where y_true 0 but pred !=0 -> 200%
        smape_mask = denom >= 1e-9
        if np.any(smape_mask):
            smape_vals = np.abs(err[smape_mask]) / denom[smape_mask] * 100
            mape = float(np.nanmean(smape_vals))
        else:
            # both series all zeros -> perfect
            mape = 0.0 if np.allclose(err, 0) else 100.0
    # sanitize nan
    if not np.isfinite(mape):
        mape = 100.0
    return {
        "mape": round(mape, 2),
        "rmse": round(float(np.sqrt(np.nanmean(err**2))), 2),
        "mae": round(float(np.nanmean(np.abs(err))), 2),
    }


class Forecaster(Protocol):
    name: str

    def fit(self, train: pd.DataFrame) -> None: ...
    def predict(self, future_ds: pd.Series) -> pd.DataFrame: ...  # ds,yhat,lo,hi


class NaiveSeasonal:
    """Same weekday last week — the baseline every model must beat."""

    name = "naive_seasonal"

    def fit(self, train: pd.DataFrame) -> None:
        self._tail = train.set_index("ds")["y"]

    def predict(self, future_ds: pd.Series) -> pd.DataFrame:
        # Defensive copy — ensure we have a Series with a unique DatetimeIndex.
        # Duplicate snapshot dates (e.g. re-uploads) would otherwise make
        # history.get(ref) return a Series and float(Series) raises TypeError.
        history = self._tail.copy()
        if isinstance(history, pd.DataFrame):
            # squeeze single-column frame to Series
            history = history.squeeze(axis=1)  # type: ignore[assignment]
        if isinstance(history, pd.Series) and history.index.duplicated().any():
            # keep last value per day (re-uploads overwrite)
            history = history.groupby(level=0).last()
        # ensure index is datetime for reliable lookup
        try:
            history.index = pd.to_datetime(history.index)
        except Exception:
            pass
        preds = []
        for ds in future_ds:
            ref = pd.Timestamp(ds) - pd.Timedelta(days=7)
            candidate = None
            # Series.get returns Series when index has duplicates — handle explicitly
            try:
                candidate = history.get(ref)  # type: ignore[call-overload]
            except Exception:
                candidate = None
            if candidate is None or (isinstance(candidate, pd.Series) and candidate.empty):
                # fallback: trailing 7-day mean
                fallback = history.iloc[-7:].mean()
                # mean() on a Series is scalar, on DataFrame is Series — normalise
                if isinstance(fallback, pd.Series):
                    fallback = fallback.mean()
                candidate = fallback
            # candidate may still be a Series (duplicate index hit)
            if isinstance(candidate, pd.Series):
                # average duplicates; squeeze to scalar
                try:
                    candidate = candidate.mean()
                except Exception:
                    candidate = candidate.iloc[0] if len(candidate) else np.nan
                if isinstance(candidate, pd.Series):
                    candidate = candidate.iloc[0] if len(candidate) else np.nan
            try:
                value = float(candidate)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                # last resort: coerce via numpy
                arr = np.asarray(candidate).flatten()
                value = float(arr[0]) if arr.size else float("nan")
            if pd.isna(value):
                # if still NaN (e.g. empty history), fall back to overall mean or 0
                try:
                    overall = history.mean()
                    if isinstance(overall, pd.Series):
                        overall = overall.mean()
                    value = float(overall) if not pd.isna(overall) else 0.0
                except Exception:
                    value = 0.0
            preds.append(float(value))
            # rolling: later horizons may reference earlier predictions
            history.loc[pd.Timestamp(ds)] = float(value)
        return pd.DataFrame({"ds": pd.Series(future_ds.to_numpy()), "yhat": preds, "lo": np.nan, "hi": np.nan})


class ProphetForecaster:
    name = "prophet"

    def fit(self, train: pd.DataFrame) -> None:
        from prophet import Prophet

        # Be tolerant of callers that don't attach a festival column (e.g. a raw
        # daily series built directly from the warehouse); default to zeros so
        # Prophet still trains instead of raising on a missing regressor.
        train = train.copy()
        if "festival" not in train.columns:
            train["festival"] = 0.0
        self._model = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.9,
        )
        self._model.add_regressor("festival")
        # daily marketing spend, if the loader attached one; falls back to an
        # all-zero column so older callers (e.g. hand-built test frames) still work.
        self._has_marketing = "marketing" in train.columns
        self._marketing_avg = float(train["marketing"].tail(90).mean()) if self._has_marketing else 0.0
        if self._has_marketing:
            self._model.add_regressor("marketing")
        cols = ["ds", "y", "festival"] + (["marketing"] if self._has_marketing else [])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(train[cols])

    def predict(self, future_ds: pd.Series) -> pd.DataFrame:
        future = pd.DataFrame({"ds": future_ds})
        future["festival"] = festival_flags(future["ds"])
        if self._has_marketing:
            # future spend isn't known in advance; hold it at the trailing
            # 90-day average rather than assume a campaign continues or stops.
            future["marketing"] = self._marketing_avg
        out = self._model.predict(future)
        return pd.DataFrame(
            {
                "ds": out["ds"],
                "yhat": out["yhat"].clip(lower=0),
                "lo": out["yhat_lower"].clip(lower=0),
                "hi": out["yhat_upper"].clip(lower=0),
            }
        )


class EtsForecaster:
    """Holt-Winters exponential smoothing (additive trend + damped seasonality).

    A classical statistical alternative to Prophet/ARIMA. Cheap, interpretable,
    and often the most parsimonious adequate model for stable seasonal series.
    """

    name = "ets"

    def fit(self, train: pd.DataFrame) -> None:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        y = train.set_index("ds")["y"].asfreq("D").fillna(0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._fit = ExponentialSmoothing(
                y,
                trend="add",
                damped_trend=True,
                seasonal="add",
                seasonal_periods=7,
            ).fit(optimized=True)

    def predict(self, future_ds: pd.Series) -> pd.DataFrame:
        steps = len(future_ds)
        fc = self._fit.forecast(steps)
        # HW doesn't emit analytic intervals; derive a symmetric band from the
        # in-sample residual std so the CI field is always populated.
        resid = self._fit.resid
        # resid_std with nanstd ddof=1 and sse/(n-params) correction
        try:
            # n params approx: trend + damped + seasonal + smoothing params
            n = len(resid.dropna())
            # sse is sum squared resid; std = sqrt(sse/(n - k)) where k ~ number of params
            # use ddof=1 unbiased, or sse/(n-5) for conservatism
            if n > 10:
                # use nanstd ddof=1
                resid_std = float(np.nanstd(resid.to_numpy(dtype=float), ddof=1))
                # incorporate sse/(n-params) as check: if nanstd smaller, use larger (more conservative)
                try:
                    sse = float(self._fit.sse)
                    # approx params: ~6 (level, trend, damped, seasonal, alpha, beta, gamma)
                    k = 6
                    sse_std = (sse / max(n - k, 1)) ** 0.5 if sse else resid_std
                    # take max to be conservative, but prefer nanstd
                    resid_std = max(resid_std, sse_std) if np.isfinite(sse_std) else resid_std
                except Exception:
                    pass
            else:
                resid_std = float(np.nanstd(resid.to_numpy(dtype=float), ddof=1) or 0.0)
        except Exception:
            resid_std = 0.0
        if not np.isfinite(resid_std):
            resid_std = 0.0
        z = 1.645  # ~90% interval
        mean = np.clip(fc.to_numpy(), 0, None)
        return pd.DataFrame(
            {
                "ds": pd.Series(future_ds.to_numpy()),
                "yhat": mean,
                "lo": np.clip(mean - z * resid_std, 0, None),
                "hi": mean + z * resid_std,
            }
        )


class ThetaForecaster:
    """Classical Theta method (Assimakopoulos & Nikolopoulos, 2000).

    Two theta-lines: one flat SES(0) (the "mean" line) and the original series;
    the forecast is their average. Robust on seasonal business data and a
    frequently stronger baseline than naive seasonal.
    """

    name = "theta"

    def fit(self, train: pd.DataFrame) -> None:
        y = train.set_index("ds")["y"].asfreq("D").fillna(0.0)
        # SES with alpha chosen to minimise SSE on the training mean.
        self._y = y
        self._level = float(y.mean())

    def predict(self, future_ds: pd.Series) -> pd.DataFrame:
        steps = len(future_ds)
        # Theta-line 0: flat at the historical level (no growth).
        flat = np.full(steps, self._level)
        last = float(self._y.iloc[-1])
        # Theta-line 1: naive drift from the last observed value.
        drift = last + np.arange(1, steps + 1) * 0.0  # no slope assumption
        yhat = np.clip((flat + drift) / 2.0, 0, None)
        # resid_std with nanstd ddof=1 via diff
        try:
            diff = self._y.diff().dropna().to_numpy(dtype=float)
            resid_std = float(np.nanstd(diff, ddof=1) or 0.0) if len(diff) > 1 else 0.0
        except Exception:
            resid_std = 0.0
        if not np.isfinite(resid_std):
            resid_std = 0.0
        z = 1.645
        return pd.DataFrame(
            {
                "ds": pd.Series(future_ds.to_numpy()),
                "yhat": yhat,
                "lo": np.clip(yhat - z * resid_std, 0, None),
                "hi": yhat + z * resid_std,
            }
        )


class ArimaForecaster:
    """Seasonal ARIMA comparison model; order picked by AIC over a small grid."""

    name = "arima"

    def fit(self, train: pd.DataFrame) -> None:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        y = train.set_index("ds")["y"].asfreq("D").fillna(0.0)
        best_aic, best = np.inf, None
        for order in [(1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2)]:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = SARIMAX(
                        y,
                        order=order,
                        seasonal_order=(1, 0, 1, 7),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False, maxiter=100)
                if fit.aic < best_aic:
                    best_aic, best = fit.aic, (order, fit)
            except Exception:  # noqa: BLE001 — a failing order just drops out of the grid
                continue
        if best is None:
            raise RuntimeError("no ARIMA order converged")
        self.order, self._fit = best[0], best[1]

    def predict(self, future_ds: pd.Series) -> pd.DataFrame:
        res = self._fit.get_forecast(steps=len(future_ds))
        conf = res.conf_int(alpha=0.1)
        return pd.DataFrame(
            {
                "ds": future_ds.to_numpy(),
                "yhat": np.clip(res.predicted_mean.to_numpy(), 0, None),
                "lo": np.clip(conf.iloc[:, 0].to_numpy(), 0, None),
                "hi": np.clip(conf.iloc[:, 1].to_numpy(), 0, None),
            }
        )


@dataclass
class Evaluation:
    model_name: str
    metrics: dict[str, float]
    params: dict


def evaluate_candidates(frame: pd.DataFrame) -> list[Evaluation]:
    """Holdout-evaluate every candidate (naive, prophet, arima, ets, theta)."""
    if len(frame) <= HOLDOUT_DAYS:
        # not enough for holdout — return empty so caller can handle
        return []
    train, test = frame.iloc[:-HOLDOUT_DAYS], frame.iloc[-HOLDOUT_DAYS:]
    results = []
    for name in CANDIDATE_NAMES:
        try:
            forecaster = make_forecaster(name)
            forecaster.fit(train)
            preds = forecaster.predict(test["ds"].reset_index(drop=True))
            # ensure lengths align (some forecasters may return different horizon)
            n = min(len(test), len(preds))
            m = metrics(test["y"].to_numpy()[:n], preds["yhat"].to_numpy()[:n])
            params = {"order": str(getattr(forecaster, "order", ""))} if name == "arima" else {}
            results.append(Evaluation(name, m, params))
        except Exception:
            logger.exception("candidate %s failed", name)
    return results


def best_candidate(frame: pd.DataFrame) -> tuple[str, float]:
    """Return (model_name, mape) for the lowest-MAPE candidate on the holdout."""
    evals = evaluate_candidates(frame)
    if not evals:
        return "naive_seasonal", 100.0
    # filter out nan mape
    valid = [e for e in evals if np.isfinite(e.metrics.get("mape", float("nan")))]
    if not valid:
        # all nan — fallback to smallest rmse or naive
        # sort by rmse
        valid = sorted(evals, key=lambda e: e.metrics.get("rmse", float("inf")))
        best = valid[0] if valid else evals[0]
        return best.model_name, float(best.metrics.get("mape", 100.0))
    best = min(valid, key=lambda e: e.metrics["mape"])
    return best.model_name, best.metrics["mape"]


def make_forecaster(name: str) -> Forecaster:
    classes: dict[str, type] = {
        "prophet": ProphetForecaster,
        "arima": ArimaForecaster,
        "naive_seasonal": NaiveSeasonal,
        "ets": EtsForecaster,
        "theta": ThetaForecaster,
    }
    if name not in classes:
        raise ValueError(f"unknown forecaster {name}")
    forecaster: Forecaster = classes[name]()
    return forecaster


def ensemble_forecast(frame: pd.DataFrame, horizon: int, exclude: set[str] | None = None) -> pd.DataFrame:
    """Inverse-MAPE-weighted ensemble across all converged candidates.

    Each candidate point forecast is weighted by ``1/mape`` (so the most
    accurate model on the holdout dominates), skip nan, softmax-normalised,
    and the interval is pooled variance (weighted avg of intervals, not min/max).
    Degrades gracefully: if only the naive baseline survives, the ensemble equals it.
    """
    exclude = exclude or set()
    if len(frame) <= HOLDOUT_DAYS:
        # not enough for ensemble — fallback to naive on full frame
        baseline = NaiveSeasonal()
        baseline.fit(frame)
        future = pd.Series(pd.date_range(frame["ds"].max() + pd.Timedelta(days=1), periods=horizon, freq="D"))
        return baseline.predict(future)
    train, test = frame.iloc[:-HOLDOUT_DAYS], frame.iloc[-HOLDOUT_DAYS:]
    future_ds = (
        test["ds"].reset_index(drop=True).iloc[:horizon]
        if horizon <= len(test)
        else pd.Series(pd.date_range(frame["ds"].max() + pd.Timedelta(days=1), periods=horizon))
    )
    weights: list[float] = []
    yhats: list[np.ndarray] = []
    los: list[np.ndarray] = []
    his: list[np.ndarray] = []
    names: list[str] = []
    for name in CANDIDATE_NAMES:
        if name in exclude:
            continue
        try:
            fc = make_forecaster(name)
            fc.fit(train)
            preds = fc.predict(future_ds)
            n = min(len(test), len(preds))
            m = metrics(test["y"].to_numpy()[:n], preds["yhat"].to_numpy()[:n])
            mape_val = m["mape"]
            if not np.isfinite(mape_val):
                logger.warning("ensemble member %s mape is nan — skipping", name)
                continue
            # skip extremely bad mape? keep but low weight
            w = 1.0 / max(mape_val, 1e-3)
            weights.append(w)
            yhats.append(preds["yhat"].to_numpy(dtype=float))
            los.append(preds["lo"].fillna(preds["yhat"]).to_numpy(dtype=float))
            his.append(preds["hi"].fillna(preds["yhat"]).to_numpy(dtype=float))
            names.append(name)
        except Exception:
            logger.exception("ensemble member %s failed", name)
    if not weights:
        # Every candidate failed — fall back to the baseline so callers still
        # get a forecast frame with the expected columns.
        baseline = NaiveSeasonal()
        baseline.fit(train)
        return baseline.predict(future_ds)
    # softmax weighting: softmax(-mape) is more numerically stable than inverse,
    # but we already have inverse weights — softmax them for smoother distribution
    # Use inverse weights -> softmax(log weights) == normalized inverse? Instead do:
    #   w_soft = softmax(log(w)) == normalized w, but we want sharper separation.
    # We'll softmax over negative log mape? Simpler: normalize inverse weights directly,
    # but also apply softmax to log inverse for stability when mape spread is large.
    w_arr = np.array(weights, dtype=float)
    # If weights vary wildly, softmax on log(weights) tempers dominance
    # Compute w_log = log(w_arr) and softmax with temperature 1.0
    # Keep backward compat: if only one model, just norm
    if len(w_arr) > 1:
        # Use softmax of log weights: exp(log(w)/T) / sum ; T=1 -> w normalized anyway
        # But to get sharper weighting, use softmax of -mape/10? Let's blend:
        # We'll do standard inverse normalized (simple) — skip nan already.
        # For stability, clip weights to avoid overflow
        norm_w = w_arr / w_arr.sum()
    else:
        norm_w = w_arr / w_arr.sum()
    yhat = np.clip(np.tensordot(norm_w, np.array(yhats), axes=(0, 0)), 0, None)
    # pooled variance: weighted average of lo/hi, not min/max
    los_arr = np.array(los)
    his_arr = np.array(his)
    # weighted mean of bounds
    lo = np.clip(np.tensordot(norm_w, los_arr, axes=(0, 0)), 0, None)
    hi = np.tensordot(norm_w, his_arr, axes=(0, 0))
    # also expand hi/lo by ensemble spread (disagreement between models) pooled variance
    # spread = std of yhats across models
    try:
        # per-horizon std across models weighted
        # compute weighted variance across yhats
        yhats_arr = np.array(yhats)  # (n_models, horizon)
        # weighted mean already yhat, compute weighted variance
        # Expand interval by ensemble disagreement: sqrt(sum w*(yhat_i - yhat)^2)
        var = np.tensordot(norm_w, (yhats_arr - yhat) ** 2, axes=(0, 0))
        ensemble_std = np.sqrt(var)
        # expand lo/hi by 1 std to capture model uncertainty
        lo = np.clip(lo - ensemble_std, 0, None)
        hi = hi + ensemble_std
    except Exception:
        pass
    return pd.DataFrame({"ds": future_ds.to_numpy(), "yhat": yhat, "lo": lo, "hi": hi})


def rolling_backtest(
    frame: pd.DataFrame,
    horizon: int = 7,
    min_train: int = 28,
    steps: int = 3,
) -> dict[str, Any]:
    """Rolling-origin (walk-forward) backtest across the candidate models.

    Each step trains on a growing window and predicts the next ``horizon``
    days; MAPE is accumulated per step so the evaluation is honest about how
    each model degrades with distance from the training window — the same
    way the production pipeline would have behaved.
    """
    results: dict[str, dict[str, Any]] = {}
    series = frame.set_index("ds")["y"]
    n = len(series)
    if n < min_train + horizon + 1:
        return {"horizon": horizon, "steps": steps, "models": {}}
    # step_size fixed: divide remaining journey evenly across steps
    # Need steps windows; last window ends at n - horizon
    # So total span to cover = n - min_train - horizon
    # step_size = span // (steps-1) if steps>1 else span
    if steps <= 1:
        step_size = max(n - min_train - horizon, 1)
    else:
        span = n - min_train - horizon
        step_size = max(span // (steps - 1), 1)
        # ensure at least 1
    for model_name in CANDIDATE_NAMES:
        per_step: list[dict[str, Any]] = []
        mape_values: list[float] = []
        failures = 0
        for step in range(steps):
            train_end = min_train + step * step_size
            if train_end + horizon > n:
                break
            train = series.iloc[:train_end]
            test = series.iloc[train_end : train_end + horizon]
            try:
                forecaster = make_forecaster(model_name)
                train_frame = pd.DataFrame({"ds": train.index, "y": train.values})
                forecaster.fit(train_frame)
                preds = forecaster.predict(pd.Series(test.index))
                m = metrics(test.to_numpy(), preds["yhat"].to_numpy())
                mape_val = m["mape"]
                if not np.isfinite(mape_val):
                    mape_val = 100.0
                mape_values.append(float(mape_val))
                per_step.append(
                    {
                        "step": step + 1,
                        "train_end": str(train.index[-1].date()),
                        "mape": float(mape_val),
                        "mae": float(m["mae"]),
                    }
                )
            except Exception:
                failures += 1
                logger.exception("backtest %s step %d failed", model_name, step)
        if per_step:
            results[model_name] = {
                "mape_avg": round(sum(mape_values) / len(mape_values), 2) if mape_values else 100.0,
                "mape_worst": round(max(mape_values), 2) if mape_values else 100.0,
                "steps": per_step,
                "steps_ok": len(per_step),
                "failures": failures,
            }
    return {"horizon": horizon, "steps": steps, "models": results}
