"""Phase 8: recommendation WHY/IMPACT/PRIORITY/ACTION + decision workflow."""

from datetime import date, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import select

from app.core.database import get_session_factory
from app.models import Insight, RecommendationFeedback, SalesTransaction
from tests.conftest import auth


@pytest.fixture
def recs_seed():
    """30 days of transactions where one channel dominates, so
    channel-boost recommendations fire with revenue-gap evidence."""

    async def _seed() -> None:
        today = date.today()
        async with get_session_factory()() as db:
            for i in range(30, 0, -1):
                d = today - timedelta(days=i)
                db.add(
                    SalesTransaction(
                        txn_date=d,
                        product_id=None,
                        quantity=10,
                        unit_price=Decimal("100"),
                        total_amount=Decimal("1000"),
                        region="Bagmati",
                        channel="retail",
                    )
                )
                db.add(
                    SalesTransaction(
                        txn_date=d,
                        product_id=None,
                        quantity=10,
                        unit_price=Decimal("100"),
                        total_amount=Decimal("100"),
                        region="Bagmati",
                        channel="wholesale",
                    )
                )
            await db.commit()

    return _seed


@pytest.mark.anyio
async def test_recommendations_carry_why_impact_priority_action(client, user_token, recs_seed):
    await recs_seed()
    _, token = user_token
    resp = await client.get("/api/v1/recommendations", headers=auth(token))
    assert resp.status_code == 200
    recs = resp.json()
    assert recs, "expected recommendations from seeded decline"
    for rec in recs:
        assert "evidence" in rec, "WHY must always be attached"
        assert rec.get("priority") in ("high", "medium", "low")
        assert rec.get("action"), "ACTION must always be attached"
    assert any(r.get("impact_estimate") for r in recs), "expected monetary impact"


@pytest.mark.anyio
async def test_generate_persists_and_decision_updates_status(client, manager_token, recs_seed):
    await recs_seed()
    _, token = manager_token

    gen = await client.post("/api/v1/recommendations/generate", headers=auth(token))
    assert gen.status_code == 200
    assert gen.json()["new"] >= 1

    hist = await client.get("/api/v1/recommendations/history", headers=auth(token))
    assert hist.status_code == 200
    rows = hist.json()
    assert rows
    assert all(r["status"] == "open" for r in rows)
    assert all(r["priority"] in ("high", "medium", "low") for r in rows)
    assert all(r["action"] for r in rows)

    target = rows[0]
    decide = await client.post(
        f"/api/v1/recommendations/{target['id']}/decide",
        json={"decision": "actioned"},
        headers=auth(token),
    )
    assert decide.status_code == 200
    assert decide.json()["status"] == "actioned"

    after = await client.get("/api/v1/recommendations/history", headers=auth(token))
    statuses = {r["id"]: r["status"] for r in after.json()}
    assert statuses[target["id"]] == "actioned"

    async with get_session_factory()() as db:
        feedback = (await db.execute(select(RecommendationFeedback))).scalars().all()
        assert any(f.rec_key == target["dedupe_key"] for f in feedback)
        insight = await db.get(Insight, target["id"])
        assert insight.status == "actioned"
        assert insight.priority in ("high", "medium", "low")
        assert insight.action
        assert insight.impact_estimate is not None


@pytest.mark.anyio
async def test_analyst_cannot_decide(client, user_token, manager_token, recs_seed):
    await recs_seed()
    _, manager_tok = manager_token
    await client.post("/api/v1/recommendations/generate", headers=auth(manager_tok))

    _, analyst_tok = user_token
    hist = await client.get("/api/v1/recommendations/history", headers=auth(analyst_tok))
    assert hist.status_code == 200
    assert hist.json()

    target = hist.json()[0]["id"]
    resp = await client.post(
        f"/api/v1/recommendations/{target}/decide",
        json={"decision": "accepted"},
        headers=auth(analyst_tok),
    )
    assert resp.status_code == 403
