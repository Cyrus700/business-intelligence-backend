"""Model registry + batch prediction: train, compare, register, persist forecasts."""

import logging
from datetime import timedelta
from typing import Any

import pandas as pd
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Forecast, MlModel
from app.services.ml import forecasting
from app.services.ml.features import discover_segments, load_series

logger = logging.getLogger(__name__)

FORECAST_TARGETS = ["revenue_daily", "orders_daily", "expenses_daily"]
DEFAULT_HORIZON = 90
# expenses are spiky with near-zero days, where MAPE degenerates — select on MAE
SELECTION_METRIC = {"expenses_daily": "mae"}
# Model type recorded for the naive baseline rows (registry CHECK allows the
# real model types only, so the baseline metrics ride along in `metrics`).


async def train_target(
    db: AsyncSession,
    target: str,
    horizon: int = DEFAULT_HORIZON,
    dimensions: dict[str, Any] | None = None,
) -> MlModel | None:
    """Evaluate candidates, register the winner (if it beats the active model),
    and write its forward forecasts.

    `dimensions` trains a segment-specific model (e.g. {"region": "Kathmandu"});
    omit for the whole-business series — same target name, disjoint segment key.
    """
    dims = dimensions or {}
    frame = await load_series(db, target, dims)
    if len(frame) < forecasting.HOLDOUT_DAYS * 2:
        logger.warning("not enough history for %s%s (%d rows)", target, dims, len(frame))
        return None

    evaluations = forecasting.evaluate_candidates(frame)
    by_name = {e.model_name: e for e in evaluations}
    baseline = by_name.get("naive_seasonal")
    candidates = [e for e in evaluations if e.model_name in ("prophet", "arima")]
    if not candidates:
        logger.error("no ML candidate converged for %s%s", target, dims)
        return None
    metric_key = SELECTION_METRIC.get(target, "mape")
    winner = min(candidates, key=lambda e: e.metrics[metric_key])

    # degradation guard: don't replace a better active model
    active = (
        await db.execute(
            select(MlModel).where(
                MlModel.target == target, MlModel.dimensions == dims, MlModel.is_active.is_(True)
            )
        )
    ).scalar_one_or_none()
    if (
        active
        and active.metrics
        and active.metrics.get(metric_key, 1e18) < winner.metrics[metric_key]
    ):
        logger.warning(
            "keeping active %s model for %s%s (%s %.2f <= new %.2f)",
            active.model_type,
            target,
            dims,
            metric_key,
            active.metrics[metric_key],
            winner.metrics[metric_key],
        )
        model = active
    else:
        version = (
            (
                await db.execute(
                    select(func.coalesce(func.max(MlModel.version), 0)).where(
                        MlModel.target == target,
                        MlModel.dimensions == dims,
                        MlModel.model_type == winner.model_name,
                    )
                )
            ).scalar_one()
        ) + 1
        await db.execute(
            update(MlModel)
            .where(MlModel.target == target, MlModel.dimensions == dims)
            .values(is_active=False)
        )
        model = MlModel(
            model_type=winner.model_name,
            target=target,
            dimensions=dims,
            version=version,
            training_rows=len(frame),
            metrics={
                **winner.metrics,
                "baseline_mape": baseline.metrics["mape"] if baseline else None,
                "candidates": {e.model_name: e.metrics for e in evaluations},
                "holdout_days": forecasting.HOLDOUT_DAYS,
            },
            params=winner.params,
            is_active=True,
        )
        db.add(model)
        await db.flush()

    # refit on the FULL history, forecast forward
    forecaster = forecasting.make_forecaster(model.model_type)
    forecaster.fit(frame)
    last = frame["ds"].max()
    future = pd.Series(
        pd.date_range(last + timedelta(days=1), periods=horizon, freq="D"), name="ds"
    )
    preds = forecaster.predict(future)

    await db.execute(delete(Forecast).where(Forecast.model_id == model.id))
    for _, row in preds.iterrows():
        db.add(
            Forecast(
                model_id=model.id,
                target=target,
                dimensions=dims,
                forecast_date=row["ds"].date(),
                horizon_days=horizon,
                yhat=round(float(row["yhat"]), 2),
                yhat_lower=None if pd.isna(row["lo"]) else round(float(row["lo"]), 2),
                yhat_upper=None if pd.isna(row["hi"]) else round(float(row["hi"]), 2),
            )
        )
    await db.commit()
    final_metrics = model.metrics or {}
    logger.info(
        "%s%s: %s v%d active (mape %.2f%% vs naive %.2f%%), %d forecast points",
        target,
        dims,
        model.model_type,
        model.version,
        final_metrics.get("mape") or float("nan"),
        final_metrics.get("baseline_mape") or float("nan"),
        horizon,
    )
    return model


# per-target cap on how many segment models get trained (top segments by
# data volume are discovered first, so this bounds retrain time predictably)
MAX_SEGMENTS_PER_TARGET = 5


async def train_all(db: AsyncSession) -> list[MlModel]:
    """Train every whole-business target, then the top segments of each."""
    models = []
    for target in FORECAST_TARGETS:
        model = await train_target(db, target)
        if model:
            models.append(model)
        for dims in (await discover_segments(db, target))[:MAX_SEGMENTS_PER_TARGET]:
            seg_model = await train_target(db, target, dimensions=dims)
            if seg_model:
                models.append(seg_model)
    return models
