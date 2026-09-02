"""Data Quality Framework tests (Phase 3 upgrade).

Covers: audit scoring on clean data, issue detection on dirty data, the
data-quality API surface, and RBAC enforcement (analyst cannot run audits).
"""

import io

from sqlalchemy import select

from app.core.database import get_session_factory
from app.models import SalesTransaction
from app.models.quality import DataQualityIssue
from tests.conftest import auth

CLEAN_CSV = (
    "date,sku,product_name,category,customer,segment,city,region,channel,"
    "quantity,unit_price,discount\n"
    "2026-06-10,BEV-001,Everest Tea 500g,Beverages,Namaste Mart,retail,Kathmandu,"
    "Bagmati,store,3,320,0\n"
    "2026-06-10,SNK-001,Wai Wai Noodles (30pk),Snacks,Everest Traders,wholesale,"
    "Biratnagar,Koshi,distributor,10,640,320\n"
    "2026-06-11,BEV-001,Everest Tea 500g,Beverages,Daraz Online Nepal,online,"
    "Kathmandu,Bagmati,online,2,320,0\n"
)

# quantity of 0 and a negative discount → validity violations
DIRTY_CSV = (
    "date,sku,product_name,category,customer,segment,city,region,channel,"
    "quantity,unit_price,discount\n"
    "2026-06-10,BEV-001,Everest Tea 500g,Beverages,Namaste Mart,retail,Kathmandu,"
    "Bagmati,store,3,320,0\n"
    "2026-06-10,SNK-001,Wai Wai Noodles (30pk),Snacks,Everest Traders,wholesale,"
    "Biratnagar,Koshi,distributor,0,640,0\n"
    "2026-06-11,BEV-001,Everest Tea 500g,Beverages,Daraz Online Nepal,online,"
    "Kathmandu,Bagmati,online,2,320,-50\n"
)


def _upload(token: str, content: str = CLEAN_CSV, name: str = "sales.csv") -> dict:
    return {
        "url": "/api/v1/uploads",
        "headers": auth(token),
        "files": {"file": (name, io.BytesIO(content.encode()), "text/csv")},
        "data": {"domain": "sales"},
    }


async def test_clean_data_scores_full(client, manager_token):
    _, token = manager_token
    kwargs = _upload(token)
    resp = await client.post(kwargs.pop("url"), **kwargs)
    assert resp.status_code == 201, resp.text

    async with get_session_factory()() as db:
        run = await _run_audit(db)
        assert run.score == 100.0
        for dim, score in run.dimensions.items():
            assert score == 100.0, dim
        assert run.issues_found == 0
        assert run.rows_checked == 5  # 3 sales rows + 2 products
        assert run.triggered_by == "manual"


async def _insert_dirty_rows() -> None:
    """Legacy dirty rows that bypassed ETL validation but are physically
    insertable under the current schema: negative discount (validity+accuracy
    violations), a future txn_date (validity), NULL product (completeness)."""
    from datetime import timedelta
    from decimal import Decimal

    from app.core.clock import business_today
    from app.models import Product

    async with get_session_factory()() as db:
        product_id = (await db.execute(select(Product.id))).scalars().first()
        db.add_all(
            [
                SalesTransaction(
                    txn_date=business_today() - timedelta(days=1),
                    product_id=product_id,
                    quantity=2,
                    unit_price=Decimal("100.00"),
                    discount=Decimal("50.00"),
                    total_amount=Decimal("50.00"),
                    channel="store",
                    region="Bagmati",
                ),
                SalesTransaction(
                    txn_date=business_today() + timedelta(days=5),
                    product_id=product_id,
                    quantity=1,
                    unit_price=Decimal("100.00"),
                    discount=Decimal("0.00"),
                    total_amount=Decimal("100.00"),
                    channel="online",
                    region="Bagmati",
                ),
                SalesTransaction(
                    txn_date=business_today() - timedelta(days=1),
                    product_id=None,
                    quantity=1,
                    unit_price=Decimal("100.00"),
                    discount=Decimal("0.00"),
                    total_amount=Decimal("100.00"),
                    channel="store",
                    region="Bagmati",
                ),
            ]
        )
        await db.commit()


async def test_dirty_data_produces_issues_and_lower_score(client, manager_token):
    _, token = manager_token
    kwargs = _upload(token)
    resp = await client.post(kwargs.pop("url"), **kwargs)
    assert resp.status_code == 201, resp.text

    await _insert_dirty_rows()

    async with get_session_factory()() as db:
        run = await _run_audit(db)
        assert run.score < 100.0
        assert run.dimensions["validity"] < 100.0
        assert run.dimensions["completeness"] < 100.0
        assert run.issues_found >= 2

        issues = (await db.execute(select(DataQualityIssue))).scalars().all()
        dims = {i.dimension for i in issues}
        assert dims >= {"validity", "completeness"}
        validity_issue = next(i for i in issues if i.dimension == "validity")
        assert validity_issue.scope_key == "domain:sales"
        assert validity_issue.status == "open"
        assert validity_issue.table_name == "sales_transactions"


async def test_overview_and_history_endpoints(client, manager_token):
    _, token = manager_token
    kwargs = _upload(token)
    await client.post(kwargs.pop("url"), **kwargs)
    run_resp = await client.post("/api/v1/data-quality/run", headers=auth(token))
    assert run_resp.status_code == 201, run_resp.text

    resp = await client.get("/api/v1/data-quality/overview", headers=auth(token))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["latest"] is not None
    assert body["latest"]["score"] == 100.0
    assert set(body["latest"]["dimensions"]) >= {
        "completeness",
        "validity",
        "consistency",
        "uniqueness",
        "timeliness",
        "accuracy",
    }
    assert len(body["trend"]) >= 1

    resp = await client.get("/api/v1/data-quality/quality/history", headers=auth(token))
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


async def test_issues_listing_and_resolution(client, manager_token):
    _, token = manager_token
    kwargs = _upload(token)
    await client.post(kwargs.pop("url"), **kwargs)
    await _insert_dirty_rows()
    run_resp = await client.post("/api/v1/data-quality/run", headers=auth(token))
    assert run_resp.status_code == 201, run_resp.text

    resp = await client.get("/api/v1/data-quality/issues", headers=auth(token))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] >= 1
    issue_id = body["items"][0]["id"]

    resp = await client.patch(
        f"/api/v1/data-quality/issues/{issue_id}",
        json={"status": "acknowledged"},
        headers=auth(token),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "acknowledged"

    resp = await client.patch(
        f"/api/v1/data-quality/issues/{issue_id}",
        json={"status": "resolved"},
        headers=auth(token),
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "resolved"


async def test_analyst_cannot_run_audit(client, user_token, manager_token):
    _, analyst = user_token
    _, token = manager_token

    resp = await client.post("/api/v1/data-quality/run", headers=auth(analyst))
    assert resp.status_code == 403, resp.text

    resp = await client.post("/api/v1/data-quality/run", headers=auth(token))
    assert resp.status_code == 201, resp.text
    assert resp.json()["triggered_by"] == "manual"


async def _run_audit(db):
    from app.services.quality.engine import run_quality_audit

    run = await run_quality_audit(db, triggered_by="manual")
    assert run is not None
    return run
