"""Phase 5 contract tests: insights, alert rules, notifications, reports."""

import uuid
from datetime import date, timedelta

from app.core.database import get_session_factory
from app.models import KpiSnapshot, Notification
from app.services.alerts.engine import evaluate_alerts
from app.services.insights.engine import generate_insights
from tests.conftest import auth, create_profile, mint_token

TODAY = date(2026, 6, 30)


async def seed_revenue(days: int = 14, recent_boost: float = 1.0) -> None:
    """Seed daily revenue snapshots ending at TODAY; last 7 days scaled by recent_boost."""
    async with get_session_factory()() as db:
        for i in range(days):
            d = TODAY - timedelta(days=days - 1 - i)
            base = 100_000.0
            if d > TODAY - timedelta(days=7):
                base *= recent_boost
            db.add(KpiSnapshot(snapshot_date=d, metric="revenue", value=base))
        await db.commit()


# ---------------------------------------------------------------- insights


async def test_insights_generate_admin_only(client, user_token):
    _, token = user_token
    resp = await client.post("/api/v1/insights/generate", headers=auth(token))
    assert resp.status_code == 403


async def test_insights_generated_and_deduped(client, admin_token):
    _, token = admin_token
    await seed_revenue(days=14, recent_boost=1.5)  # +50% shift, above 15% threshold

    first = await client.post("/api/v1/insights/generate", headers=auth(token))
    assert first.status_code == 202
    assert first.json()["created"] >= 1

    listing = await client.get("/api/v1/insights", headers=auth(token))
    assert listing.status_code == 200
    titles = [i["title"] for i in listing.json()]
    assert any("Revenue" in t for t in titles)

    # Re-running must not duplicate anything (dedupe_key unique constraint).
    second = await client.post("/api/v1/insights/generate", headers=auth(token))
    assert second.json()["created"] == 0


async def test_insight_pin_requires_manager(client, user_token, admin_token):
    _, admin = admin_token
    await seed_revenue(days=14, recent_boost=1.5)
    async with get_session_factory()() as db:
        await generate_insights(db)
    listing = (await client.get("/api/v1/insights", headers=auth(admin))).json()
    insight_id = listing[0]["id"]

    _, analyst = user_token
    resp = await client.patch(f"/api/v1/insights/{insight_id}/pin", headers=auth(analyst))
    assert resp.status_code == 403

    resp = await client.patch(f"/api/v1/insights/{insight_id}/pin", headers=auth(admin))
    assert resp.status_code == 200
    assert resp.json()["is_pinned"] is True


# ---------------------------------------------------------------- alert rules


async def test_alert_rules_manager_rbac(client, user_token):
    _, analyst = user_token
    resp = await client.get("/api/v1/alert-rules", headers=auth(analyst))
    assert resp.status_code == 403


async def test_alert_rule_threshold_required_unless_anomaly(client, admin_token):
    _, token = admin_token
    resp = await client.post(
        "/api/v1/alert-rules",
        headers=auth(token),
        json={"name": "no threshold", "metric": "revenue", "condition": "gt"},
    )
    assert resp.status_code == 422

    resp = await client.post(
        "/api/v1/alert-rules",
        headers=auth(token),
        json={"name": "anomaly watch", "metric": "revenue", "condition": "anomaly_detected"},
    )
    assert resp.status_code == 201


async def test_alert_evaluation_fires_and_respects_cooldown(client, admin_token):
    admin_profile, token = admin_token
    await seed_revenue(days=14)

    resp = await client.post(
        "/api/v1/alert-rules",
        headers=auth(token),
        json={
            "name": "weekly revenue floor",
            "metric": "revenue",
            "condition": "gt",
            "threshold": 500_000,  # 7 × 100k = 700k > 500k → fires
            "window_days": 7,
            "roles_notified": ["admin"],
        },
    )
    assert resp.status_code == 201

    async with get_session_factory()() as db:
        created = await evaluate_alerts(db, today=TODAY)
    assert created == 1

    notifications = await client.get("/api/v1/notifications", headers=auth(token))
    assert notifications.status_code == 200
    assert len(notifications.json()) == 1
    assert "weekly revenue floor" in notifications.json()[0]["title"]

    # Second evaluation within the 23h cooldown must not re-notify.
    async with get_session_factory()() as db:
        created = await evaluate_alerts(db, today=TODAY)
    assert created == 0


# ---------------------------------------------------------------- notifications


async def test_notifications_own_rows_only(client, user_token, admin_token):
    analyst_profile, analyst = user_token
    admin_profile, admin = admin_token
    async with get_session_factory()() as db:
        mine = Notification(id=uuid.uuid4(), user_id=analyst_profile.id, title="for analyst")
        theirs = Notification(id=uuid.uuid4(), user_id=admin_profile.id, title="for admin")
        db.add_all([mine, theirs])
        await db.commit()
        mine_id, theirs_id = mine.id, theirs.id

    listing = (await client.get("/api/v1/notifications", headers=auth(analyst))).json()
    assert [n["title"] for n in listing] == ["for analyst"]

    resp = await client.patch(f"/api/v1/notifications/{theirs_id}/read", headers=auth(analyst))
    assert resp.status_code == 404

    resp = await client.patch(f"/api/v1/notifications/{mine_id}/read", headers=auth(analyst))
    assert resp.status_code == 200
    assert resp.json()["is_read"] is True


# ---------------------------------------------------------------- reports


async def test_report_generate_requires_manager(client, user_token):
    _, analyst = user_token
    resp = await client.post(
        "/api/v1/reports/generate",
        headers=auth(analyst),
        json={"period_start": "2026-06-01", "period_end": "2026-06-30"},
    )
    assert resp.status_code == 403


async def test_report_pdf_roundtrip(client):
    manager = await create_profile("manager")
    token = mint_token(manager.id, "manager")
    await seed_revenue(days=30)

    resp = await client.post(
        "/api/v1/reports/generate",
        headers=auth(token),
        json={"period_start": "2026-06-01", "period_end": "2026-06-30", "format": "pdf"},
    )
    assert resp.status_code == 201
    report_id = resp.json()["id"]

    download = await client.get(f"/api/v1/reports/{report_id}/download", headers=auth(token))
    assert download.status_code == 200
    assert download.content.startswith(b"%PDF")
    assert "attachment" in download.headers["content-disposition"]


async def test_report_xlsx_roundtrip(client):
    manager = await create_profile("manager")
    token = mint_token(manager.id, "manager")
    await seed_revenue(days=30)

    resp = await client.post(
        "/api/v1/reports/generate",
        headers=auth(token),
        json={"period_start": "2026-06-01", "period_end": "2026-06-30", "format": "xlsx"},
    )
    assert resp.status_code == 201

    download = await client.get(
        f"/api/v1/reports/{resp.json()['id']}/download", headers=auth(token)
    )
    assert download.status_code == 200
    assert download.content.startswith(b"PK")  # xlsx = zip container
