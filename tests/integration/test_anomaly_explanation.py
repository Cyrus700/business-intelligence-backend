"""Anomaly explanation + resolution lifecycle (Phase 7)."""

from datetime import date, timedelta
from decimal import Decimal

from app.core.database import get_session_factory
from app.models import Anomaly, KpiSnapshot, SalesTransaction
from app.services.ml.anomaly import scan_target
from tests.conftest import auth


async def seed_sales_history() -> date:
    """60 days of steady revenue, ending with a one-region spike day."""
    today = date.today()
    spike_day = today
    spike_region = "Sudurpashchim"
    async with get_session_factory()() as db:
        # Baseline: ~10k/day from a few regions. Revenue metric is the
        # whole-business series, so kpi_snapshots aggregate across regions.
        for i in range(60, 0, -1):
            d = today - timedelta(days=i)
            value = 10000 + (i % 7) * 100
            db.add(
                KpiSnapshot(
                    snapshot_date=d, metric="revenue", dimensions={}, value=value
                )
            )
            db.add(
                SalesTransaction(
                    txn_date=d,
                    product_id=None,
                    quantity=10,
                    unit_price=Decimal(value) / 10,
                    total_amount=Decimal(value),
                    region="Bagmati",
                    channel="retail",
                )
            )
        # Spike day: one region contributes 180k of the 200k total.
        db.add(
            KpiSnapshot(
                snapshot_date=spike_day, metric="revenue", dimensions={}, value=200000
            )
        )
        db.add(
            SalesTransaction(
                txn_date=spike_day,
                product_id=None,
                quantity=180,
                unit_price=Decimal(1000),
                total_amount=Decimal(180000),
                region=spike_region,
                channel="wholesale",
            )
        )
        db.add(
            SalesTransaction(
                txn_date=spike_day,
                product_id=None,
                quantity=20,
                unit_price=Decimal(1000),
                total_amount=Decimal(20000),
                region="Bagmati",
                channel="retail",
            )
        )
        await db.commit()
    return spike_day


async def test_anomaly_detection_stores_contributor_explanation(client, user_token):
    spike_day = await seed_sales_history()
    async with get_session_factory()() as db:
        created = await scan_target(db, "revenue_daily")
    assert created >= 1

    _, token = user_token
    resp = await client.get("/api/v1/anomalies", headers=auth(token))
    assert resp.status_code == 200, resp.text
    anomalies = resp.json()
    spike = next((a for a in anomalies if a["context"].get("date") == spike_day.isoformat()), None)
    assert spike is not None, "spike day should be flagged"
    explanation = spike["explanation"]
    assert explanation is not None
    assert explanation["day"] == spike_day.isoformat()
    assert explanation["baseline_days"] == 28
    assert explanation["primary"]["dimension"] == "region"
    assert explanation["primary"]["key"] == "Sudurpashchim"
    assert explanation["primary"]["share_pct"] > 30
    assert len(explanation["contributors"]) >= 2
    assert any(
        c["dimension"] == "channel" and c["key"] == "wholesale"
        for c in explanation["contributors"]
    )


async def test_anomaly_resolution_lifecycle(client, manager_token):
    """Resolved status records who/when; reopening clears the trail."""
    _, token = manager_token
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
        anomaly_id = anomaly.id

    resp = await client.patch(
        f"/api/v1/anomalies/{anomaly_id}",
        json={"status": "resolved"},
        headers=auth(token),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "resolved"
    assert body["resolved_at"] is not None
    assert body["resolved_by"] is not None

    resp = await client.patch(
        f"/api/v1/anomalies/{anomaly_id}",
        json={"status": "open"},
        headers=auth(token),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "open"
    assert body["resolved_at"] is None
    assert body["resolved_by"] is None


async def test_anomaly_rejection_of_invalid_status(client, manager_token):
    _, token = manager_token
    async with get_session_factory()() as db:
        anomaly = Anomaly(metric="revenue", observed_value=1, expected_value=1)
        db.add(anomaly)
        await db.commit()
        anomaly_id = anomaly.id

    resp = await client.patch(
        f"/api/v1/anomalies/{anomaly_id}",
        json={"status": "closed"},
        headers=auth(token),
    )
    assert resp.status_code == 422
