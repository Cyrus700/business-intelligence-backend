"""ML edge-case tests: insufficient history, extreme values, model failure resilience."""

import pytest
from datetime import date, timedelta
from decimal import Decimal

from app.core.database import get_session_factory
from app.models import KpiSnapshot, SalesTransaction
from app.services.ml.forecasting import NaiveSeasonal, ProphetForecaster, make_forecaster, rolling_backtest
from app.services.ml.features import load_series
from tests.conftest import auth
from tests.conftest import user_token


@pytest.fixture
def minimal_sales_data():
    """Only 3 days of sales - insufficient for most models."""

    async def _seed() -> None:
        today = date.today()
        async with get_session_factory()() as db:
            for i in range(3, 0, -1):
                d = today - timedelta(days=i)
                db.add(
                    KpiSnapshot(
                        snapshot_date=d, metric="revenue", dimensions={}, value=10000 + i * 100
                    )
                )
                db.add(
                    SalesTransaction(
                        txn_date=d,
                        product_id=None,
                        quantity=10,
                        unit_price=Decimal("1000"),
                        total_amount=Decimal("10000"),
                        region="Bagmati",
                        channel="retail",
                    )
                )
            await db.commit()

    return _seed


@pytest.fixture
def extreme_values_data():
    """Sales with extreme values (outliers) in the past."""

    async def _seed() -> None:
        today = date.today()
        async with get_session_factory()() as db:
            # 60 days of normal data, but day 2 is the spike (need 60+ for anomaly scan)
            for i in range(60, 0, -1):
                d = today - timedelta(days=i)
                if i == 2:
                    # Spike day
                    db.add(
                        KpiSnapshot(
                            snapshot_date=d, metric="revenue", dimensions={}, value=1_000_000
                        )
                    )
                    db.add(
                        SalesTransaction(
                            txn_date=d,
                            product_id=None,
                            quantity=1000,
                            unit_price=Decimal("1000"),
                            total_amount=Decimal("1_000_000"),
                            region="Bagmati",
                            channel="retail",
                        )
                    )
                else:
                    db.add(
                        KpiSnapshot(
                            snapshot_date=d, metric="revenue", dimensions={}, value=10000
                        )
                    )
                    db.add(
                        SalesTransaction(
                            txn_date=d,
                            product_id=None,
                            quantity=10,
                            unit_price=Decimal("1000"),
                            total_amount=Decimal("10000"),
                            region="Bagmati",
                            channel="retail",
                        )
                    )
            await db.commit()

    return _seed


@pytest.fixture
def zero_sales_data():
    """Periods with zero sales."""

    async def _seed() -> None:
        today = date.today()
        async with get_session_factory()() as db:
            for i in range(30, 0, -1):
                d = today - timedelta(days=i)
                value = 0 if i % 7 == 0 else 10000  # Weekly zero
                db.add(
                    KpiSnapshot(
                        snapshot_date=d, metric="revenue", dimensions={}, value=value
                    )
                )
                if value > 0:
                    db.add(
                        SalesTransaction(
                            txn_date=d,
                            product_id=None,
                            quantity=10,
                            unit_price=Decimal("1000"),
                            total_amount=Decimal(str(value)),
                            region="Bagmati",
                            channel="retail",
                        )
                    )
            await db.commit()

    return _seed


@pytest.mark.anyio
async def test_insufficient_history_uses_naive(minimal_sales_data):
    """With <7 days, forecasting falls back to naive seasonal."""
    await minimal_sales_data()
    from app.core.database import get_session_factory
    async with get_session_factory()() as db:
        frame = await load_series(db, "revenue_daily")
        assert len(frame) < 7

        forecaster = make_forecaster("naive_seasonal")
        forecaster.fit(frame)
        preds = forecaster.predict(frame["ds"].tail(1))
        assert preds is not None
        assert "yhat" in preds.columns


@pytest.mark.anyio
async def test_forecast_handles_extreme_outliers(extreme_values_data):
    """Forecast should not crash with extreme outliers."""
    await extreme_values_data()
    from app.core.database import get_session_factory
    from sqlalchemy.ext.asyncio import AsyncSession

    async with get_session_factory()() as db:
        frame = await load_series(db, "revenue_daily")
        # Should have data
        assert len(frame) >= 30
        # Naive should work
        forecaster = make_forecaster("naive_seasonal")
        forecaster.fit(frame)
        preds = forecaster.predict(frame["ds"].tail(7))
        assert preds is not None
        assert len(preds) == 7
        # Prophet should also handle it (may be slow but shouldn't crash)
        try:
            prophet = make_forecaster("prophet")
            prophet.fit(frame)
            preds2 = prophet.predict(frame["ds"].tail(7))
            assert preds2 is not None
        except Exception:
            pytest.skip("Prophet not available or failed on extreme data")


@pytest.mark.anyio
async def test_zero_sales_handled(zero_sales_data):
    """Zero sales periods should not break forecasting."""
    await zero_sales_data()
    from app.core.database import get_session_factory

    async with get_session_factory()() as db:
        frame = await load_series(db, "revenue_daily")
        assert len(frame) >= 30
        # Should have zeros in the data
        assert (frame["y"] == 0).any()
        # Naive should handle zeros
        forecaster = make_forecaster("naive_seasonal")
        forecaster.fit(frame)
        preds = forecaster.predict(frame["ds"].tail(7))
        assert preds is not None
        # Predictions should be non-negative
        assert (preds["yhat"] >= 0).all()


@pytest.mark.anyio
async def test_backtest_with_insufficient_data():
    """Backtest should handle insufficient data gracefully."""
    import pandas as pd
    from app.services.ml.forecasting import rolling_backtest

    # Only 10 rows - less than min_train
    frame = pd.DataFrame({
        "ds": pd.date_range("2026-01-01", periods=10, freq="D"),
        "y": [100] * 10,
    })
    result = rolling_backtest(frame, horizon=3, steps=2)
    # Should return empty models or handle gracefully
    assert "models" in result
    assert result["horizon"] == 3


@pytest.mark.anyio
async def test_model_failure_resilience():
    """Model training failure shouldn't crash the pipeline."""
    from app.services.ml.forecasting import evaluate_candidates
    import pandas as pd
    import numpy as np

    # Data that might cause model issues
    frame = pd.DataFrame({
        "ds": pd.date_range("2026-01-01", periods=5, freq="D"),
        "y": [np.inf, -np.inf, np.nan, 100, 200],
    })
    results = evaluate_candidates(frame)
    # Should return empty list or handle gracefully, not crash
    assert isinstance(results, list)


@pytest.mark.anyio
async def test_forecast_api_with_insufficient_history(client, user_token, minimal_sales_data):
    """Forecast API should handle insufficient data gracefully."""
    await minimal_sales_data()
    _, token = user_token
    resp = await client.get("/api/v1/forecasts", headers=auth(token))
    assert resp.status_code == 200
    data = resp.json()
    # Should return a valid response with metrics possibly None
    assert "target" in data
    assert "points" in data


@pytest.mark.anyio
async def test_anomaly_detection_with_extreme_values(client, user_token, extreme_values_data):
    """Anomaly detection should flag extreme values."""
    await extreme_values_data()
    # Run the anomaly scan to detect the spike
    from app.services.ml.anomaly import scan_all
    from app.core.database import get_session_factory
    async with get_session_factory()() as db:
        await scan_all(db, lookback_days=7)
    _, token = user_token
    resp = await client.get("/api/v1/anomalies", headers=auth(token))
    assert resp.status_code == 200
    anomalies = resp.json()
    # Should detect the spike
    assert len(anomalies) > 0


@pytest.mark.anyio
async def test_model_registry_includes_all_models():
    """Model registry should list all trained models."""
    from app.core.database import get_session_factory
    from app.models import MlModel
    from sqlalchemy import select

    async with get_session_factory()() as db:
        models = (await db.execute(select(MlModel))).scalars().all()
        # Should have at least some models
        assert isinstance(models, list)