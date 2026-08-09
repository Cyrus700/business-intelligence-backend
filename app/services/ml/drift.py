"""Drift-triggered retrain: compare each active model's recent live accuracy
against its own training-time holdout MAPE, and retrain early if it has
degraded past a tolerance band (docs/05-ml-plan.md weekly-retrain is the
floor; this catches degradation sooner).

Every check is recorded as a ModelDrift row, triggered or not, so "why did
this retrain early" is always answerable from the audit trail.
"""

import logging
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Forecast, ModelDrift
from app.models.ml import MlModel
from app.services.ml.features import load_series
from app.services.ml.forecasting import metrics as compute_metrics
from app.services.ml.registry import train_target

logger = logging.getLogger(__name__)

WINDOW_DAYS = 7
# how much worse than training-time MAPE we tolerate before forcing a retrain
DEGRADATION_FACTOR = 1.5
MIN_THRESHOLD_MAPE = 10.0


async def check_drift(
    db: AsyncSession, model: MlModel, window_days: int = WINDOW_DAYS
) -> ModelDrift:
    """Score `model`'s stored forecasts against actuals for the last
    `window_days` and record (and possibly act on) the result."""
    holdout_mape = (model.metrics or {}).get("mape")
    threshold = max((holdout_mape or 0) * DEGRADATION_FACTOR, MIN_THRESHOLD_MAPE)

    frame = await load_series(db, model.target, model.dimensions)
    live_mape = None
    measured_on = frame["ds"].max().date() if len(frame) else None
    if len(frame) and measured_on:
        window_start = measured_on - timedelta(days=window_days - 1)
        actual = frame[(frame["ds"].dt.date >= window_start) & (frame["ds"].dt.date <= measured_on)]
        rows = (
            await db.execute(
                select(Forecast).where(
                    Forecast.model_id == model.id,
                    Forecast.forecast_date >= window_start,
                    Forecast.forecast_date <= measured_on,
                )
            )
        ).scalars().all()
        by_date = {r.forecast_date: float(r.yhat) for r in rows}
        paired = [
            (row["y"], by_date[row["ds"].date()])
            for _, row in actual.iterrows()
            if row["ds"].date() in by_date
        ]
        if paired:
            import numpy as np

            y_true = np.array([p[0] for p in paired])
            y_pred = np.array([p[1] for p in paired])
            live_mape = compute_metrics(y_true, y_pred)["mape"]

    triggered = live_mape is not None and live_mape > threshold
    drift = ModelDrift(
        model_id=model.id,
        target=model.target,
        measured_on=measured_on or model.trained_at.date(),
        window_days=window_days,
        live_mape=live_mape,
        holdout_mape=holdout_mape,
        threshold_mape=threshold,
        triggered=triggered,
    )
    db.add(drift)
    await db.commit()

    if triggered:
        logger.warning(
            "drift triggered retrain: %s%s live_mape=%.2f > threshold=%.2f",
            model.target, model.dimensions or "", live_mape, threshold,
        )
        await train_target(db, model.target, dimensions=model.dimensions)
    return drift


async def check_all(db: AsyncSession, window_days: int = WINDOW_DAYS) -> list[ModelDrift]:
    models = (await db.execute(select(MlModel).where(MlModel.is_active.is_(True)))).scalars().all()
    results = []
    for model in models:
        try:
            results.append(await check_drift(db, model, window_days))
        except Exception:
            logger.exception("drift check failed for %s%s", model.target, model.dimensions or "")
    return results
