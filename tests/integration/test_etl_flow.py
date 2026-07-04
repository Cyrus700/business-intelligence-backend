import io

from sqlalchemy import func, select, text

from app.core.database import get_session_factory
from app.models import KpiSnapshot, Product, SalesTransaction
from tests.conftest import auth

CSV = (
    "date,sku,product_name,category,customer,segment,city,region,channel,"
    "quantity,unit_price,discount\n"
    "2026-06-10,BEV-001,Everest Tea 500g,Beverages,Namaste Mart,retail,Kathmandu,"
    "Bagmati,store,3,320,0\n"
    "2026-06-10,SNK-001,Wai Wai Noodles (30pk),Snacks,Everest Traders,wholesale,"
    "Biratnagar,Koshi,distributor,10,640,320\n"
    "2026-06-11,BEV-001,Everest Tea 500g,Beverages,Daraz Online Nepal,online,"
    "Kathmandu,Bagmati,online,2,320,0\n"
    "bad-date,BEV-001,Everest Tea 500g,Beverages,Namaste Mart,retail,Kathmandu,"
    "Bagmati,store,1,320,0\n"
)


def _upload(token: str, content: str = CSV, name: str = "sales.csv", domain: str = "sales"):
    return {
        "url": "/api/v1/uploads",
        "headers": auth(token),
        "files": {"file": (name, io.BytesIO(content.encode()), "text/csv")},
        "data": {"domain": domain},
    }


async def test_upload_loads_and_reports_rejects(client, user_token):
    _, token = user_token
    kwargs = _upload(token)
    resp = await client.post(kwargs.pop("url"), **kwargs)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["status"] == "loaded"
    assert body["row_count"] == 4
    assert body["error_report"]["loaded"] == 3
    assert body["error_report"]["rejected"] == 1
    assert body["error_report"]["details"][0]["reason"].startswith("unparseable date")

    async with get_session_factory()() as db:
        assert (await db.execute(select(func.count(SalesTransaction.id)))).scalar() == 3
        skus = {p.sku for p in (await db.execute(select(Product))).scalars()}
        assert skus == {"BEV-001", "SNK-001"}


async def test_upload_is_idempotent(client, user_token):
    _, token = user_token
    for expected_loaded, expected_dupes in ((3, 0), (0, 3)):
        kwargs = _upload(token)
        resp = await client.post(kwargs.pop("url"), **kwargs)
        report = resp.json()["error_report"]
        assert report["loaded"] == expected_loaded
        assert report["skipped_duplicates"] == expected_dupes

    async with get_session_factory()() as db:
        assert (await db.execute(select(func.count(SalesTransaction.id)))).scalar() == 3


async def test_upload_rebuilds_kpi_snapshots(client, user_token):
    _, token = user_token
    kwargs = _upload(token)
    await client.post(kwargs.pop("url"), **kwargs)
    async with get_session_factory()() as db:
        rows = (
            (
                await db.execute(
                    select(KpiSnapshot).where(
                        KpiSnapshot.metric == "revenue", KpiSnapshot.dimensions == {}
                    )
                )
            )
            .scalars()
            .all()
        )
        by_date = {r.snapshot_date.isoformat(): float(r.value) for r in rows}
    # day 1: 3*320 + (10*640 - 320) = 960 + 6080 = 7040 ; day 2: 640
    assert by_date == {"2026-06-10": 7040.0, "2026-06-11": 640.0}


async def test_upload_missing_columns_fails_cleanly(client, user_token):
    _, token = user_token
    kwargs = _upload(token, content="date,sku\n2026-06-10,X\n")
    resp = await client.post(kwargs.pop("url"), **kwargs)
    assert resp.status_code == 422
    assert "missing required columns" in resp.json()["detail"]


async def test_expense_upload(client, user_token):
    _, token = user_token
    csv = "date,category,amount,department\n2026-06-01,rent,185000,operations\n"
    kwargs = _upload(token, content=csv, name="expenses.csv", domain="finance")
    resp = await client.post(kwargs.pop("url"), **kwargs)
    assert resp.status_code == 201
    assert resp.json()["error_report"]["loaded"] == 1


async def test_etl_jobs_visible_to_admin_only(client, user_token, admin_token):
    _, token = user_token
    _, admin_tok = admin_token
    kwargs = _upload(token)
    await client.post(kwargs.pop("url"), **kwargs)

    resp = await client.get("/api/v1/etl/jobs", headers=auth(token))
    assert resp.status_code == 403

    resp = await client.get("/api/v1/etl/jobs", headers=auth(admin_tok))
    assert resp.status_code == 200
    jobs = resp.json()
    assert jobs[0]["trigger"] == "upload"
    assert jobs[0]["status"] == "succeeded"
    assert jobs[0]["rows_loaded"] == 3


async def test_data_source_crud_and_postgres_pull(client, admin_token):
    _, token = admin_token
    # analystless CRUD: create a postgres source pointing at our own test DB
    async with get_session_factory()() as db:
        await db.execute(
            text(
                "CREATE TABLE IF NOT EXISTS legacy_pos AS "
                "SELECT '2026-06-12'::text AS date, 'BEV-002'::text AS sku, "
                "2 AS quantity, 560 AS unit_price"
            )
        )
        await db.commit()

    from tests.conftest import TEST_DB_URL

    resp = await client.post(
        "/api/v1/data-sources",
        headers=auth(token),
        json={
            "name": "Legacy POS DB",
            "kind": "postgres",
            "target_domain": "sales",
            "config": {"dsn": TEST_DB_URL, "query": "SELECT * FROM legacy_pos"},
        },
    )
    assert resp.status_code == 201, resp.text
    source_id = resp.json()["id"]

    run = await client.post(f"/api/v1/etl/run/{source_id}", headers=auth(token))
    assert run.status_code == 200, run.text
    assert run.json()["rows_loaded"] == 1

    async with get_session_factory()() as db:
        await db.execute(text("DROP TABLE legacy_pos"))
        await db.commit()


async def test_paused_source_cannot_run(client, admin_token):
    _, token = admin_token
    resp = await client.post(
        "/api/v1/data-sources",
        headers=auth(token),
        json={
            "name": "Paused src",
            "kind": "rest_api",
            "target_domain": "sales",
            "config": {"url": "https://x.example.com"},
        },
    )
    source_id = resp.json()["id"]
    await client.patch(
        f"/api/v1/data-sources/{source_id}", headers=auth(token), json={"status": "paused"}
    )
    run = await client.post(f"/api/v1/etl/run/{source_id}", headers=auth(token))
    assert run.status_code == 409
