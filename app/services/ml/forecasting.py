"""Forecasting: Prophet (primary) vs ARIMA vs naive seasonal baseline.

Evaluation protocol (docs/05-ml-plan.md): time-based split — train on all but
the last HOLDOUT_DAYS, score on the holdout, always against the naive seasonal
baseline (same weekday, previous week). The winner is registered in ml_models
and its batch predictions land in the forecasts table.
"""

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
        history = self._tail.copy()
        preds = []
        for ds in future_ds:
            ref = ds - pd.Timedelta(days=7)
            value = float(history.get(ref, history.iloc[-7:].mean()))
            preds.append(value)
            history[ds] = value  # rolling: later horizons reference predictions
        return pd.DataFrame({"ds": future_ds, "yhat": preds, "lo": np.nan, "hi": np.nan})


class ProphetForecaster:
    name = "prophet"

    def fit(self, train: pd.DataFrame) -> None:
        from prophet import Prophet

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
    """Holdout-evaluate naive, prophet, arima on the given series."""
    train, test = frame.iloc[:-HOLDOUT_DAYS], frame.iloc[-HOLDOUT_DAYS:]
    results = []
    for forecaster in (NaiveSeasonal(), ProphetForecaster(), ArimaForecaster()):
        try:
            forecaster.fit(train)
            preds = forecaster.predict(test["ds"].reset_index(drop=True))
            m = metrics(test["y"].to_numpy(), preds["yhat"].to_numpy())
            params = (
                {"order": str(getattr(forecaster, "order", ""))}
                if forecaster.name == "arima"
                else {}
            )
            results.append(Evaluation(forecaster.name, m, params))
        except Exception:
            logger.exception("candidate %s failed", forecaster.name)
    return results


def make_forecaster(name: str) -> Forecaster:
    classes: dict[str, type] = {
        "prophet": ProphetForecaster,
        "arima": ArimaForecaster,
        "naive_seasonal": NaiveSeasonal,
    }
    forecaster: Forecaster = classes[name]()
    return forecaster


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
    for model_name in ("naive_seasonal", "prophet", "arima"):
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
