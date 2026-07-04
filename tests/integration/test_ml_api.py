"""ML endpoint contract tests. Model rows are seeded directly (training is
covered by unit tests + scripts/evaluate_ml.py; API tests stay fast)."""

from datetime import date, timedelta

from app.core.database import get_session_factory
from app.models import Anomaly, Forecast, MlModel
from tests.conftest import auth


async def seed_model_with_forecasts() -> MlModel:
    async with get_session_factory()() as db:
        model = MlModel(
            model_type="prophet",
            target="revenue_daily",
            version=1,
            training_rows=1000,
            metrics={"mape": 37.35, "baseline_mape": 52.58, "holdout_days": 90},
            is_active=True,
        )
        db.add(model)
        await db.flush()
        for i in range(30):
            db.add(
                Forecast(
                    model_id=model.id,
                    target="revenue_daily",
                    forecast_date=date.today() + timedelta(days=i + 1),
                    horizon_days=30,
                    yhat=100000 + i * 100,
                    yhat_lower=90000,
                    yhat_upper=110000,
                )
            )
        await db.commit()
        return model


async def seed_anomaly() -> Anomaly:
    async with get_session_factory()() as db:
        anomaly = Anomaly(
            metric="revenue",
            observed_value=210000,
            expected_value=84000,
            deviation_score=6.2,
            severity="high",
            context={"date": "2026-06-19", "direction": "above", "pct_deviation": 150.0},
        )
        db.add(anomaly)
        await db.commit()
        return anomaly


async def test_forecast_404_when_untrained(client, user_token):
    _, token = user_token
    resp = await client.get("/api/v1/forecasts", headers=auth(token))
    assert resp.status_code == 404
    assert "retrain" in resp.json()["detail"]


async def test_forecast_returns_points_and_metrics(client, user_token):
    _, token = user_token
    await seed_model_with_forecasts()
    resp = await client.get("/api/v1/forecasts", headers=auth(token), params={"horizon": 14})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_type"] == "prophet"
    assert body["metrics"]["baseline_mape"] == 52.58
    assert len(body["points"]) == 14
    assert body["points"][0]["yhat"] == 100000.0


async def test_accuracy_lists_active_models(client, user_token):
    _, token = user_token
    await seed_model_with_forecasts()
    resp = await client.get("/api/v1/forecasts/accuracy", headers=auth(token))
    assert resp.status_code == 200
    assert resp.json()[0]["metrics"]["mape"] == 37.35


async def test_retrain_is_admin_only(client, user_token):
    _, token = user_token
    resp = await client.post("/api/v1/forecasts/retrain", headers=auth(token))
    assert resp.status_code == 403


async def test_anomaly_listing_and_filtering(client, user_token):
    _, token = user_token
    await seed_anomaly()
    resp = await client.get("/api/v1/anomalies", headers=auth(token), params={"severity": "high"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["context"]["pct_deviation"] == 150.0


async def test_anomaly_ack_requires_manager(client, user_token, admin_token):
    _, analyst_tok = user_token
    admin_profile, admin_tok = admin_token
    anomaly = await seed_anomaly()

    resp = await client.patch(
        f"/api/v1/anomalies/{anomaly.id}",
        headers=auth(analyst_tok),
        json={"status": "acknowledged"},
    )
    assert resp.status_code == 403

    resp = await client.patch(
        f"/api/v1/anomalies/{anomaly.id}",
        headers=auth(admin_tok),
        json={"status": "acknowledged"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "acknowledged"


async def test_trends_endpoint(client, user_token, seeded_kpis=None):
    _, token = user_token
    resp = await client.get("/api/v1/trends", headers=auth(token))
    # empty warehouse in this test → not enough history
    assert resp.status_code == 404
