"""Diagnostic analytics tests (Phase 5 upgrade).

Verifies the contribution decomposition: sum of member deltas ≈ total delta,
primary/secondary factors are populated, and the API contract holds.
"""

import io

from tests.conftest import auth

CSV = (
    "date,sku,product_name,category,customer,segment,city,region,channel,"
    "quantity,unit_price,discount\n"
    # period 1 (June): Bagmati contributes 5 * 1000 = 5000, Koshi 2000
    "2026-06-10,BEV-001,Everest Tea 500g,Beverages,Namaste Mart,retail,Kathmandu,"
    "Bagmati,store,5,1000,0\n"
    "2026-06-10,SNK-001,Wai Wai Noodles (30pk),Snacks,Everest Traders,wholesale,"
    "Biratnagar,Koshi,distributor,2,1000,0\n"
    # period 2 (July): Bagmati jumps to 9000, Koshi falls to 500
    "2026-07-10,BEV-001,Everest Tea 500g,Beverages,Namaste Mart,retail,Kathmandu,"
    "Bagmati,store,9,1000,0\n"
    "2026-07-10,SNK-001,Wai Wai Noodles (30pk),Snacks,Everest Traders,wholesale,"
    "Biratnagar,Koshi,distributor,1,500,0\n"
)


def _upload(token: str) -> dict:
    return {
        "url": "/api/v1/uploads",
        "headers": auth(token),
        "files": {"file": ("sales.csv", io.BytesIO(CSV.encode()), "text/csv")},
        "data": {"domain": "sales"},
    }


async def test_diagnostic_decomposition(client, manager_token):
    _, token = manager_token
    kwargs = _upload(token)
    resp = await client.post(kwargs.pop("url"), **kwargs)
    assert resp.status_code == 201, resp.text

    resp = await client.get(
        "/api/v1/diagnostics/change",
        params={
            "from": "2026-07-01",
            "to": "2026-07-31",
            "metric": "revenue",
            "dimensions": "region,channel",
        },
        headers=auth(token),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # June: 7000 total → July: 9500 total → delta +2500, +35.7%
    assert body["previous"] == 7000
    assert body["current"] == 9500
    assert body["delta"] == 2500
    assert body["change_pct"] == 35.7
    assert body["direction"] == "up"

    # Region decomposition: Bagmati +4000, Koshi −1500 → sum == delta
    regions = {m["key"]: m for m in body["dimensions"]["region"]["members"]}
    assert abs((regions["Bagmati"]["delta"] + regions["Koshi"]["delta"]) - body["delta"]) < 0.01
    assert regions["Bagmati"]["contribution_pct"] == 160.0  # +4000 / 2500
    assert regions["Koshi"]["contribution_pct"] == -60.0

    # Primary factor must be Bagmati (region), the largest absolute driver
    summary = body["summary"]
    assert summary["primary_contributor"]["key"] == "Bagmati"
    assert summary["primary_factor"] is not None
    assert "Bagmati" in summary["primary_factor"]

    # Channel: distributor dropped out entirely — key survives in the matrix
    channels = {m["key"]: m for m in body["dimensions"]["channel"]["members"]}
    assert set(channels) == {"store", "distributor"}
    assert channels["distributor"]["current"] == 500
    assert channels["distributor"]["previous"] == 2000


async def test_diagnostic_decline_direction(client, manager_token):
    """A revenue *decline* flips the summary direction and drag list."""
    _, token = manager_token
    csv = (
        "date,sku,product_name,category,customer,segment,city,region,channel,"
        "quantity,unit_price,discount\n"
        "2026-06-10,BEV-001,Everest Tea 500g,Beverages,Namaste Mart,retail,Kathmandu,"
        "Bagmati,store,10,1000,0\n"
        "2026-07-10,BEV-001,Everest Tea 500g,Beverages,Namaste Mart,retail,Kathmandu,"
        "Bagmati,store,2,1000,0\n"
    )
    kwargs = {
        "url": "/api/v1/uploads",
        "headers": auth(token),
        "files": {"file": ("sales.csv", io.BytesIO(csv.encode()), "text/csv")},
        "data": {"domain": "sales"},
    }
    resp = await client.post(kwargs.pop("url"), **kwargs)
    assert resp.status_code == 201, resp.text

    resp = await client.get(
        "/api/v1/diagnostics/change",
        params={"from": "2026-07-01", "to": "2026-07-31", "dimensions": "region"},
        headers=auth(token),
    )
    body = resp.json()
    assert body["direction"] == "down"
    assert body["summary"]["direction_word"] == "decline"
    assert body["summary"]["primary_contributor"]["key"] == "Bagmati"
    assert body["dimensions"]["region"]["net_contribution"] == 100.0


async def test_diagnostic_expense_metric(client, manager_token):
    """Expense_total decomposes by category, not sales dimensions."""
    _, token = manager_token
    resp = await client.get(
        "/api/v1/diagnostics/change",
        params={
            "from": "2026-07-01",
            "to": "2026-07-31",
            "metric": "expense_total",
            "dimensions": "category",
        },
        headers=auth(token),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["current"] == 0
    assert "category" in resp.json()["dimensions"]