"""Analytics endpoint tests against a small fixture dataset with hand-computed
expected aggregates (docs/06-testing-strategy.md Phase 3 obligations)."""

import io

import pytest

from tests.conftest import auth

# 2026-06-10: BEV(3*320=960 Bagmati/store) + SNK(10*640-320=6080 Koshi/distributor)
# 2026-06-11: BEV(2*320=640 Bagmati/online)
# 2026-05-15: BEV(5*320=1600 Bagmati/store)   <- previous-period comparator
CSV = (
    "date,sku,product_name,category,customer,segment,city,region,channel,"
    "quantity,unit_price,discount\n"
    "2026-06-10,BEV-001,Everest Tea,Beverages,Namaste Mart,retail,Kathmandu,Bagmati,store,3,320,0\n"
    "2026-06-10,SNK-001,Wai Wai,Snacks,Everest Traders,wholesale,Biratnagar,Koshi,distributor,10,640,320\n"
    "2026-06-11,BEV-001,Everest Tea,Beverages,Daraz Online,online,Kathmandu,Bagmati,online,2,320,0\n"
    "2026-05-15,BEV-001,Everest Tea,Beverages,Namaste Mart,retail,Kathmandu,Bagmati,store,5,320,0\n"
)
EXPENSES_CSV = "date,category,amount\n2026-06-05,rent,5000\n2026-06-20,marketing,2500\n"
INVENTORY_CSV = (
    "date,sku,quantity_on_hand,reorder_level\n"
    "2026-06-30,BEV-001,500,120\n2026-06-30,SNK-001,100,120\n"
)

RANGE = {"from": "2026-06-01", "to": "2026-06-30"}


@pytest.fixture
async def seeded(client, manager_token):
    _, token = manager_token
    for content, name, domain in (
        (CSV, "sales.csv", "sales"),
        (EXPENSES_CSV, "expenses.csv", "finance"),
        (INVENTORY_CSV, "inventory.csv", "inventory"),
    ):
        resp = await client.post(
            "/api/v1/uploads",
            headers=auth(token),
            files={"file": (name, io.BytesIO(content.encode()), "text/csv")},
            data={"domain": domain},
        )
        assert resp.status_code == 201, resp.text
    return token


async def test_kpi_summary_values_and_period_change(client, seeded):
    resp = await client.get("/api/v1/kpis/summary", headers=auth(seeded), params=RANGE)
    assert resp.status_code == 200
    cards = {c["metric"]: c for c in resp.json()["cards"]}
    assert cards["revenue"]["value"] == 7680.0  # 960+6080+640
    assert cards["orders"]["value"] == 3
    assert cards["expense_total"]["value"] == 7500.0
    # previous 30-day window (May 2..May 31) has the 1600 sale
    assert cards["revenue"]["previous_value"] == 1600.0
    assert cards["revenue"]["change_pct"] == 380.0


async def test_kpi_summary_with_region_filter(client, seeded):
    resp = await client.get(
        "/api/v1/kpis/summary", headers=auth(seeded), params={**RANGE, "region": "Bagmati"}
    )
    cards = {c["metric"]: c for c in resp.json()["cards"]}
    assert cards["revenue"]["value"] == 1600.0  # 960 + 640
    assert "expense_total" not in cards  # expenses carry no sales dimensions


async def test_timeseries_daily_revenue(client, seeded):
    resp = await client.get(
        "/api/v1/kpis/timeseries",
        headers=auth(seeded),
        params={**RANGE, "metric": "revenue", "granularity": "day"},
    )
    points = {p["period"]: p["value"] for p in resp.json()["points"]}
    assert points == {"2026-06-10": 7040.0, "2026-06-11": 640.0}


async def test_sales_by_product_with_share(client, seeded):
    resp = await client.get("/api/v1/sales/by-product", headers=auth(seeded), params=RANGE)
    rows = resp.json()
    assert rows[0]["key"] == "Wai Wai"
    assert rows[0]["revenue"] == 6080.0
    assert rows[0]["share_pct"] == 79.2
    assert rows[1]["sku"] == "BEV-001"


async def test_drilldown_transactions_by_sku(client, seeded):
    resp = await client.get(
        "/api/v1/sales/transactions",
        headers=auth(seeded),
        params={**RANGE, "sku": "BEV-001"},
    )
    body = resp.json()
    assert body["total"] == 2
    assert all(t["sku"] == "BEV-001" for t in body["items"])
    assert body["items"][0]["txn_date"] == "2026-06-11"  # newest first


async def test_pnl_requires_manager(client, seeded, user_token, admin_token):
    # `seeded` uploads as a manager (uploads are manager+), so the denial case
    # needs its own analyst token rather than the seeding one.
    _, analyst_tok = user_token
    resp = await client.get("/api/v1/finance/pnl", headers=auth(analyst_tok), params=RANGE)
    assert resp.status_code == 403

    _, admin_tok = admin_token
    resp = await client.get("/api/v1/finance/pnl", headers=auth(admin_tok), params=RANGE)
    assert resp.status_code == 200
    row = resp.json()[0]
    assert row["revenue"] == 7680.0
    assert row["expenses"] == 7500.0


async def test_inventory_levels_flags_reorder(client, seeded):
    resp = await client.get("/api/v1/inventory/levels", headers=auth(seeded))
    rows = {r["sku"]: r for r in resp.json()}
    assert rows["SNK-001"]["below_reorder"] is True
    assert rows["BEV-001"]["below_reorder"] is False

    resp = await client.get(
        "/api/v1/inventory/levels", headers=auth(seeded), params={"below_reorder": "true"}
    )
    assert [r["sku"] for r in resp.json()] == ["SNK-001"]


async def test_analytics_requires_auth(client):
    assert (await client.get("/api/v1/kpis/summary")).status_code == 401
