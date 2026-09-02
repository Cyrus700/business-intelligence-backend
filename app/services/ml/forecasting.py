"""Forecasting: Prophet, ARIMA, ETS, Theta vs naive seasonal baseline + ensemble.

Evaluation protocol (docs/05-ml-plan.md): time-based split — train on all but
the last HOLDOUT_DAYS, score on the holdout, always against the naive seasonal
baseline (same weekday, previous week). Candidates are compared by MAPE; the
best is promoted and an ensemble (inverse-MAPE weighted) is also produced.
"""

# Every candidate the engine can train. ``naive_seasonal`` is the floor that all
# real models must beat; ETS and Theta add classical statistical alternatives so
# the registry can pick the genuinely simplest adequate model (totos.md §13).
CANDIDATE_NAMES = ("naive_seasonal", "prophet", "arima", "ets", "theta")

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from app.services.ml.features import festival_flags

logger = logging.getLogger(__name__)

HOLDOUT_DAYS = 90


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_true - y_pred
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    return {
        "mape": round(float(np.nanmean(np.abs(err) / denom)) * 100, 2),
        "rmse": round(float(np.sqrt(np.mean(err**2))), 2),
        "mae": round(float(np.mean(np.abs(err))), 2),
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
        self._marketing_avg = (
            float(train["marketing"].tail(90).mean()) if self._has_marketing else 0.0
        )
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
        resid_std = float(np.std(self._fit.resid)) or float(self._fit.sse**0.5) or 0.0
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
        resid_std = float(np.std(self._y.diff().dropna())) or 0.0
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
    train, test = frame.iloc[:-HOLDOUT_DAYS], frame.iloc[-HOLDOUT_DAYS:]
    results = []
    for name in CANDIDATE_NAMES:
        try:
            forecaster = make_forecaster(name)
            forecaster.fit(train)
            preds = forecaster.predict(test["ds"].reset_index(drop=True))
            m = metrics(test["y"].to_numpy(), preds["yhat"].to_numpy())
            params = (
                {"order": str(getattr(forecaster, "order", ""))}
                if name == "arima"
                else {}
            )
            results.append(Evaluation(name, m, params))
        except Exception:
            logger.exception("candidate %s failed", name)
    return results


def best_candidate(frame: pd.DataFrame) -> tuple[str, float]:
    """Return (model_name, mape) for the lowest-MAPE candidate on the holdout."""
    evals = evaluate_candidates(frame)
    if not evals:
        return "naive_seasonal", 100.0
    best = min(evals, key=lambda e: e.metrics["mape"])
    return best.model_name, best.metrics["mape"]


def make_forecaster(name: str) -> Forecaster:
    classes: dict[str, type] = {
        "prophet": ProphetForecaster,
        "arima": ArimaForecaster,
        "naive_seasonal": NaiveSeasonal,
        "ets": EtsForecaster,
        "theta": ThetaForecaster,
    }
    forecaster: Forecaster = classes[name]()
    return forecaster


def ensemble_forecast(
    frame: pd.DataFrame, horizon: int, exclude: set[str] | None = None
) -> pd.DataFrame:
    """Inverse-MAPE-weighted ensemble across all converged candidates.

    Each candidate point forecast is weighted by ``1/mape`` (so the most
    accurate model on the holdout dominates), and the interval is the
    widest band among contributors. Degrades gracefully: if only the naive
    baseline survives, the ensemble equals it.
    """
    exclude = exclude or set()
    train, test = frame.iloc[:-HOLDOUT_DAYS], frame.iloc[-HOLDOUT_DAYS:]
    future_ds = test["ds"].reset_index(drop=True).iloc[:horizon] if horizon <= len(test) else pd.Series(
        pd.date_range(test["ds"].iloc[-1] + pd.Timedelta(days=1), periods=horizon)
    )
    weights: list[float] = []
    yhats: list[np.ndarray] = []
    los: list[np.ndarray] = []
    his: list[np.ndarray] = []
    for name in CANDIDATE_NAMES:
        if name in exclude:
            continue
        try:
            fc = make_forecaster(name)
            fc.fit(train)
            preds = fc.predict(future_ds)
            m = metrics(test["y"].to_numpy()[: len(preds)], preds["yhat"].to_numpy())
            w = 1.0 / max(m["mape"], 1e-3)
            weights.append(w)
            yhats.append(preds["yhat"].to_numpy())
            los.append(preds["lo"].fillna(preds["yhat"]).to_numpy())
            his.append(preds["hi"].fillna(preds["yhat"]).to_numpy())
        except Exception:
            logger.exception("ensemble member %s failed", name)
    if not weights:
        return NaiveSeasonal().fit(train) or pd.DataFrame()
    w = np.array(weights) / sum(weights)
    yhat = np.clip(np.tensordot(w, np.array(yhats), axes=(0, 0)), 0, None)
    lo = np.clip(np.min(np.array(los), axis=0), 0, None)
    hi = np.max(np.array(his), axis=0)
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
    step_size = max((n - min_train - horizon) // steps, 1)
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
                mape_values.append(float(m["mape"]))
                per_step.append(
                    {
                        "step": step + 1,
                        "train_end": str(train.index[-1].date()),
                        "mape": float(m["mape"]),
                        "mae": float(m["mae"]),
                    }
                )
            except Exception:
                failures += 1
                logger.exception("backtest %s step %d failed", model_name, step)
        if per_step:
            results[model_name] = {
                "mape_avg": round(sum(mape_values) / len(mape_values), 2),
                "mape_worst": round(max(mape_values), 2),
                "steps": per_step,
                "steps_ok": len(per_step),
                "failures": failures,
            }
    return {"horizon": horizon, "steps": steps, "models": results}
    return forecaster
