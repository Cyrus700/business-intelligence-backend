"""ML endpoints: /forecasts, /anomalies, /trends (Phase 4)."""

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal
from uuid import UUID

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession, get_current_user, require_role
from app.core.clock import business_now
from app.models import Anomaly, Forecast, MlModel

router = APIRouter(tags=["ml"], dependencies=[Depends(get_current_user)])


class ForecastPoint(BaseModel):
    forecast_date: date
    yhat: float
    yhat_lower: float | None
    yhat_upper: float | None


class ForecastOut(BaseModel):
    target: str
    model_type: str
    model_version: int
    generated_at: datetime
    metrics: dict[str, Any] | None
    points: list[ForecastPoint]


class AccuracyOut(BaseModel):
    target: str
    model_type: str
    version: int
    trained_at: datetime
    training_rows: int | None
    metrics: dict[str, Any] | None


class AnomalyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    detected_at: datetime
    metric: str
    observed_value: Decimal
    expected_value: Decimal | None
    deviation_score: Decimal | None
    severity: str
    status: str
    context: dict[str, Any] | None


class AnomalyUpdate(BaseModel):
    status: Literal["acknowledged", "dismissed", "open"]


class TrendOut(BaseModel):
    metric: str
    window_days: int
    direction: str
    weekly_change_pct: float
    strength_r: float
    current_level: float


async def _active_model(
    db, target: str, dimensions: dict[str, Any] | None = None
) -> MlModel | None:
    return (
        await db.execute(
            select(MlModel).where(
                MlModel.target == target,
                MlModel.dimensions == (dimensions or {}),
                MlModel.is_active.is_(True),
            )
        )
    ).scalar_one_or_none()


def _roll30_summary(df: pd.DataFrame) -> dict[str, Any] | None:
    if len(df) < 7:
        return None
    tail = df.tail(30) if len(df) >= 30 else df
    avg = tail["y"].mean()
    return {"mape": round(float(tail["y"].std() / avg * 100), 2) if avg > 0 else None}


@router.get("/forecasts", response_model=ForecastOut)
async def get_forecast(
    db: DbSession,
    target: str = "revenue_daily",
    horizon: int = Query(30, ge=1, le=90),
    region: str | None = None,
    channel: str | None = None,
    category: str | None = None,
) -> ForecastOut:
    from app.services.ml.features import load_series
    from app.services.ml.forecasting import NaiveSeasonal

    dims: dict[str, Any] = {}
    if region:
        dims["region"] = region
    if channel:
        dims["channel"] = channel
    if category:
        dims["category"] = category

    model = await _active_model(db, target, dims)
    if model is not None:
        rows = (
            (
                await db.execute(
                    select(Forecast)
                    .where(Forecast.model_id == model.id)
                    .order_by(Forecast.forecast_date)
                    .limit(horizon)
                )
            )
            .scalars()
            .all()
        )
        if rows:
            return ForecastOut(
                target=target,
                model_type=model.model_type,
                model_version=model.version,
                generated_at=rows[0].generated_at if rows else model.trained_at,
                metrics=model.metrics,
                points=[
                    ForecastPoint(
                        forecast_date=r.forecast_date,
                        yhat=float(r.yhat),
                        yhat_lower=None if r.yhat_lower is None else float(r.yhat_lower),
                        yhat_upper=None if r.yhat_upper is None else float(r.yhat_upper),
                    )
                    for r in rows
                ],
            )

    frame = await load_series(db, target, dims)
    if len(frame) < 7:
        return ForecastOut(
            target=target,
            model_type="naive_seasonal",
            model_version=0,
            generated_at=business_now().replace(tzinfo=None),
            metrics={"mape": None, "note": "insufficient data (need ≥ 7 days)"},
            points=[],
        )

    forecaster = NaiveSeasonal()
    forecaster.fit(frame)
    future = pd.date_range(
        start=frame["ds"].max() + pd.Timedelta(days=1), periods=horizon, freq="D"
    )
    preds = forecaster.predict(pd.Series(future))
    m = _roll30_summary(frame)
    return ForecastOut(
        target=target,
        model_type="naive_seasonal",
        model_version=0,
        generated_at=business_now().replace(tzinfo=None),
        metrics=m,
        points=[
            ForecastPoint(
                forecast_date=row["ds"].date(),
                yhat=float(row["yhat"]),
                yhat_lower=None if pd.isna(row["lo"]) else float(row["lo"]),
                yhat_upper=None if pd.isna(row["hi"]) else float(row["hi"]),
            )
            for _, row in preds.iterrows()
        ],
    )


@router.get("/forecasts/accuracy", response_model=list[AccuracyOut])
async def forecast_accuracy(db: DbSession) -> list[AccuracyOut]:
    models = (await db.execute(select(MlModel).where(MlModel.is_active.is_(True)))).scalars().all()
    return [
        AccuracyOut(
            target=m.target,
            model_type=m.model_type,
            version=m.version,
            trained_at=m.trained_at,
            training_rows=m.training_rows,
            metrics=m.metrics,
        )
        for m in models
    ]


@router.post(
    "/forecasts/retrain",
    dependencies=[Depends(require_role("admin"))],
    status_code=status.HTTP_202_ACCEPTED,
)
async def retrain(db: DbSession) -> dict[str, Any]:
    """Synchronous retrain of all targets (takes ~30-60s; acceptable for admin use)."""
    from app.services.ml.registry import train_all

    models = await train_all(db)
    return {
        "retrained": [
            {
                "target": m.target,
                "model": m.model_type,
                "version": m.version,
                "mape": (m.metrics or {}).get("mape"),
            }
            for m in models
        ]
    }


@router.get("/anomalies", response_model=list[AnomalyOut])
async def list_anomalies(
    db: DbSession,
    status_filter: str | None = Query(None, alias="status"),
    severity: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
) -> list[AnomalyOut]:
    stmt = select(Anomaly).order_by(Anomaly.detected_at.desc())
    if status_filter:
        stmt = stmt.where(Anomaly.status == status_filter)
    if severity:
        stmt = stmt.where(Anomaly.severity == severity)
    stmt = stmt.offset((page - 1) * page_size).limit(page_size)
    rows = (await db.execute(stmt)).scalars().all()
    return [AnomalyOut.model_validate(r) for r in rows]


@router.patch(
    "/anomalies/{anomaly_id}",
    response_model=AnomalyOut,
    dependencies=[Depends(require_role("manager"))],
)
async def update_anomaly(
    anomaly_id: UUID, body: AnomalyUpdate, db: DbSession, user: CurrentUser
) -> AnomalyOut:
    anomaly = await db.get(Anomaly, anomaly_id)
    if anomaly is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Anomaly not found")
    anomaly.status = body.status
    anomaly.acknowledged_by = user.id if body.status != "open" else None
    await db.commit()
    await db.refresh(anomaly)
    return AnomalyOut.model_validate(anomaly)


@router.get("/trends", response_model=TrendOut)
async def get_trend(
    db: DbSession,
    metric: Literal["revenue", "orders", "expenses"] = "revenue",
    window_days: int = Query(90, ge=14, le=365),
) -> TrendOut:
    from app.services.ml.trends import trend_summary

    summary = await trend_summary(db, metric, window_days)
    if summary is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not enough history for a trend")
    return TrendOut(**summary)
