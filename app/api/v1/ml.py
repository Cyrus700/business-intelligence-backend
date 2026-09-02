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
    explanation: dict[str, Any] | None
    resolved_at: datetime | None
    resolved_by: UUID | None


class AnomalyUpdate(BaseModel):
    status: Literal["acknowledged", "dismissed", "open", "resolved"]


class TrendOut(BaseModel):
    metric: str
    window_days: int
    direction: str
    weekly_change_pct: float
    strength_r: float
    current_level: float


class ModelOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    model_type: str
    target: str
    dimensions: dict[str, Any]
    version: int
    trained_at: datetime
    training_rows: int | None
    metrics: dict[str, Any] | None
    params: dict[str, Any] | None
    is_active: bool
    activated_at: datetime | None
    retired_at: datetime | None
    dataset_start: date | None = None
    dataset_end: date | None = None


class BacktestStep(BaseModel):
    step: int
    train_end: date
    mape: float
    mae: float


class BacktestModel(BaseModel):
    mape_avg: float
    mape_worst: float
    steps: list[BacktestStep]
    steps_ok: int
    failures: int


class BacktestOut(BaseModel):
    horizon: int
    steps: int
    models: dict[str, BacktestModel]


async def _active_model(
    db, target: str, dimensions: dict[str, Any] | None = None, org_id=None
) -> MlModel | None:
    q = select(MlModel).where(
        MlModel.target == target,
        MlModel.dimensions == (dimensions or {}),
        MlModel.is_active.is_(True),
    )
    if org_id is not None:
        q = q.where(MlModel.org_id == org_id)
    return (await db.execute(q)).scalar_one_or_none()


def _roll30_summary(df: pd.DataFrame) -> dict[str, Any] | None:
    if len(df) < 7:
        return None
    tail = df.tail(30) if len(df) >= 30 else df
    avg = tail["y"].mean()
    return {"mape": round(float(tail["y"].std() / avg * 100), 2) if avg > 0 else None}


@router.get("/forecasts", response_model=ForecastOut)
async def get_forecast(
    db: DbSession,
    user: CurrentUser,
    target: str = "revenue_daily",
    horizon: int = Query(30, ge=1, le=90),
    region: str | None = None,
    channel: str | None = None,
    category: str | None = None,
) -> ForecastOut:
    from app.api.deps import is_super_admin
    from app.services.ml.features import load_series
    from app.services.ml.forecasting import NaiveSeasonal

    dims: dict[str, Any] = {}
    if region:
        dims["region"] = region
    if channel:
        dims["channel"] = channel
    if category:
        dims["category"] = category
    org_id = None if is_super_admin(user) else user.org_id

    model = await _active_model(db, target, dims, org_id=org_id)
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

    frame = await load_series(db, target, dims, org_id=org_id)
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
async def forecast_accuracy(db: DbSession, user: CurrentUser) -> list[AccuracyOut]:
    from app.api.deps import is_super_admin

    q = select(MlModel).where(MlModel.is_active.is_(True))
    if not is_super_admin(user) and user.org_id is not None:
        q = q.where(MlModel.org_id == user.org_id)
    models = (await db.execute(q)).scalars().all()
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


@router.get("/models", response_model=list[ModelOut])
async def model_registry(db: DbSession, user: CurrentUser) -> list[ModelOut]:
    """Model registry (Phase 6) — every trained model with lifecycle dates,
    metrics, params, and the dataset range it was trained on."""
    from sqlalchemy import text
    from app.api.deps import is_super_admin

    q = select(MlModel).order_by(MlModel.trained_at.desc(), MlModel.version.desc())
    if not is_super_admin(user) and user.org_id is not None:
        q = q.where(MlModel.org_id == user.org_id)
    models = ((await db.execute(q)).scalars().all())

    dataset_range: dict[str, tuple[date | None, date | None]] = {}
    for m in models:
        if m.target in dataset_range:
            continue
        if user.org_id is not None and not is_super_admin(user):
            row = (
                await db.execute(
                    text(
                        "SELECT MIN(snapshot_date) AS s, MAX(snapshot_date) AS e "
                        "FROM kpi_snapshots WHERE metric = :m AND org_id = :org"
                    ),
                    {"m": m.target.replace("_daily", "") if m.target.endswith("_daily") else m.target, "org": str(user.org_id)},
                )
            ).first()
        else:
            row = (
                await db.execute(
                    text(
                        "SELECT MIN(snapshot_date) AS s, MAX(snapshot_date) AS e "
                        "FROM kpi_snapshots WHERE metric = :m"
                    ),
                    {"m": m.target.replace("_daily", "") if m.target.endswith("_daily") else m.target},
                )
            ).first()
        dataset_range[m.target] = (row.s if row else None, row.e if row else None)

    out = []
    for m in models:
        start, end = dataset_range.get(m.target, (None, None))
        out.append(
            ModelOut(
                id=m.id,
                model_type=m.model_type,
                target=m.target,
                dimensions=m.dimensions,
                version=m.version,
                trained_at=m.trained_at,
                training_rows=m.training_rows,
                metrics=m.metrics,
                params=m.params,
                is_active=m.is_active,
                activated_at=m.activated_at,
                retired_at=m.retired_at,
                dataset_start=start,
                dataset_end=end,
            )
        )
    return out


@router.post(
    "/models/{model_id}/retire",
    response_model=ModelOut,
    dependencies=[Depends(require_role("admin"))],
)
async def retire_model(model_id: UUID, db: DbSession, user: CurrentUser) -> ModelOut:
    """Retire a model version from the registry (admin); the newest active
    version stays in production."""
    from app.api.deps import is_super_admin

    model = await db.get(MlModel, model_id)
    if model is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model not found")
    if not is_super_admin(user) and model.org_id != user.org_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model not found")
    if not model.is_active:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Model already retired")
    model.is_active = False
    model.retired_at = business_now()
    await db.commit()
    await db.refresh(model)
    return ModelOut(
        id=model.id,
        model_type=model.model_type,
        target=model.target,
        dimensions=model.dimensions,
        version=model.version,
        trained_at=model.trained_at,
        training_rows=model.training_rows,
        metrics=model.metrics,
        params=model.params,
        is_active=model.is_active,
        activated_at=model.activated_at,
        retired_at=model.retired_at,
    )


@router.get("/backtest", response_model=BacktestOut)
async def backtest(
    db: DbSession,
    user: CurrentUser,
    horizon: int = Query(7, ge=1, le=30),
    steps: int = Query(3, ge=1, le=6),
) -> BacktestOut:
    """Rolling-origin backtest of naive vs prophet vs arima on the revenue
    series — honest walk-forward MAPE so the dashboard can show which model
    holds up beyond the training window (Phase 6)."""
    from app.api.deps import is_super_admin
    from app.services.ml.features import load_series
    from app.services.ml.forecasting import rolling_backtest

    org_id = None if is_super_admin(user) else user.org_id
    frame = await load_series(db, "revenue_daily", org_id=org_id)
    if len(frame) < 60:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "Need at least 60 days of history for a meaningful backtest",
        )
    result = rolling_backtest(frame, horizon=horizon, steps=steps)
    return BacktestOut(**result)


@router.post(
    "/forecasts/retrain",
    dependencies=[Depends(require_role("admin"))],
    status_code=status.HTTP_202_ACCEPTED,
)
async def retrain(db: DbSession, user: CurrentUser) -> dict[str, Any]:
    """Synchronous retrain of all targets (takes ~30-60s; acceptable for admin use)."""
    from app.services.ml.registry import train_all
    from app.api.deps import is_super_admin

    org_id = None if is_super_admin(user) else user.org_id
    models = await train_all(db, org_id=org_id)
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
    user: CurrentUser,
    status_filter: str | None = Query(None, alias="status"),
    severity: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
) -> list[AnomalyOut]:
    from app.api.deps import is_super_admin

    stmt = select(Anomaly).order_by(Anomaly.detected_at.desc())
    if not is_super_admin(user) and user.org_id is not None:
        stmt = stmt.where(Anomaly.org_id == user.org_id)
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
    from app.api.deps import is_super_admin

    anomaly = await db.get(Anomaly, anomaly_id)
    if anomaly is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Anomaly not found")
    if not is_super_admin(user) and anomaly.org_id != user.org_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Anomaly not found")
    anomaly.status = body.status
    anomaly.acknowledged_by = user.id if body.status != "open" else None
    if body.status == "resolved":
        anomaly.resolved_at = anomaly.resolved_at or business_now()
        anomaly.resolved_by = anomaly.resolved_by or user.id
    else:
        anomaly.resolved_at = None
        anomaly.resolved_by = None
    await db.commit()
    await db.refresh(anomaly)
    return AnomalyOut.model_validate(anomaly)


@router.get("/trends", response_model=TrendOut)
async def get_trend(
    db: DbSession,
    user: CurrentUser,
    metric: Literal["revenue", "orders", "expenses"] = "revenue",
    window_days: int = Query(90, ge=14, le=365),
) -> TrendOut:
    from app.api.deps import is_super_admin
    from app.services.ml.trends import trend_summary

    org_id = None if is_super_admin(user) else user.org_id
    summary = await trend_summary(db, metric, window_days, org_id=org_id)
    if summary is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not enough history for a trend")
    return TrendOut(**summary)
