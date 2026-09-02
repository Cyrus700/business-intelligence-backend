"""Day-level date filtering and data-coverage reporting.

The fixture data is the same shape as test_analytics.py: three sales days and
two expense days with hand-computed totals, so a filter that drifts by one day
— or a bucket that silently converts through UTC — fails loudly here.
"""

import io

import pytest

from tests.conftest import auth

# 2026-06-10: 3*320 = 960  and  10*640-320 = 6080  → 7040
# 2026-06-11: 2*320 = 640
# 2026-06-12: 4*320 = 1280
CSV = (
    "date,sku,product_name,category,customer,segment,city,region,channel,"
    "quantity,unit_price,discount\n"
    "2026-06-10,BEV-001,Everest Tea,Beverages,Namaste Mart,retail,"
    "Kathmandu,Bagmati,store,3,320,0\n"
    "2026-06-10,SNK-001,Wai Wai,Snacks,Everest Traders,wholesale,"
    "Biratnagar,Koshi,distributor,10,640,320\n"
    "2026-06-11,BEV-001,Everest Tea,Beverages,Daraz Online,online,"
    "Kathmandu,Bagmati,online,2,320,0\n"
    "2026-06-12,BEV-001,Everest Tea,Beverages,Namaste Mart,retail,"
    "Kathmandu,Bagmati,store,4,320,0\n"
)
EXPENSES_CSV = "date,category,amount\n2026-06-10,rent,5000\n2026-06-12,marketing,2500\n"


@pytest.fixture
async def seeded(client, manager_token):
    _, token = manager_token
    uploads = ((CSV, "sales.csv", "sales"), (EXPENSES_CSV, "exp.csv", "finance"))
    for content, name, domain in uploads:
        resp = await client.post(
            "/api/v1/uploads",
            headers=auth(token),
            files={"file": (name, io.BytesIO(content.encode()), "text/csv")},
            data={"domain": domain},
        )
        assert resp.status_code == 201, resp.text
    return token


async def test_single_day_filter_returns_only_that_day(client, seeded):
    """from == to must select exactly one day, inclusive at both ends."""
    resp = await client.get(
        "/api/v1/kpis/summary",
        headers=auth(seeded),
        params={"from": "2026-06-10", "to": "2026-06-10"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["period_start"] == "2026-06-10"
    assert body["period_end"] == "2026-06-10"
    cards = {c["metric"]: c for c in body["cards"]}
    assert cards["revenue"]["value"] == 7040.0
    assert cards["orders"]["value"] == 2


async def test_single_day_previous_period_is_the_day_before(client, seeded):
    """A 1-day window compares against the single preceding day, not 30."""
    resp = await client.get(
        "/api/v1/kpis/summary",
        headers=auth(seeded),
        params={"from": "2026-06-11", "to": "2026-06-11"},
    )
    cards = {c["metric"]: c for c in resp.json()["cards"]}
    assert cards["revenue"]["value"] == 640.0
    assert cards["revenue"]["previous_value"] == 7040.0  # the 10th


async def test_daily_buckets_are_not_shifted_by_timezone(client, seeded):
    """Regression: date_trunc returns TIMESTAMPTZ and used to shift a day back.

    With the database on Asia/Kathmandu (+05:45) the driver converted each
    bucket to UTC, so 2026-06-10 was reported as 2026-06-09.
    """
    resp = await client.get(
        "/api/v1/kpis/timeseries",
        headers=auth(seeded),
        params={
            "from": "2026-06-01",
            "to": "2026-06-30",
            "metric": "revenue",
            "granularity": "day",
        },
    )
    points = {p["period"]: p["value"] for p in resp.json()["points"]}
    assert points == {"2026-06-10": 7040.0, "2026-06-11": 640.0, "2026-06-12": 1280.0}


async def test_expense_daily_buckets_are_not_shifted(client, seeded):
    resp = await client.get(
        "/api/v1/kpis/timeseries",
        headers=auth(seeded),
        params={
            "from": "2026-06-01",
            "to": "2026-06-30",
            "metric": "expense_total",
            "granularity": "day",
        },
    )
    points = {p["period"]: p["value"] for p in resp.json()["points"]}
    assert points == {"2026-06-10": 5000.0, "2026-06-12": 2500.0}


async def test_seven_day_window_is_inclusive_of_both_ends(client, seeded):
    resp = await client.get(
        "/api/v1/kpis/summary",
        headers=auth(seeded),
        params={"from": "2026-06-06", "to": "2026-06-12"},
    )
    cards = {c["metric"]: c for c in resp.json()["cards"]}
    assert cards["revenue"]["value"] == 7040.0 + 640.0 + 1280.0


async def test_monthly_pnl_month_is_not_shifted(client, seeded, admin_token):
    _, token = admin_token
    resp = await client.get(
        "/api/v1/finance/pnl",
        headers=auth(token),
        params={"from": "2026-06-01", "to": "2026-06-30"},
    )
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    # must be June, not 2026-05-31 rolled back through UTC
    assert rows[0]["month"] == "2026-06-01"
    assert rows[0]["revenue"] == 8960.0
    assert rows[0]["expenses"] == 7500.0


async def test_window_outside_loaded_data_returns_zero_not_error(client, seeded):
    resp = await client.get(
        "/api/v1/kpis/summary",
        headers=auth(seeded),
        params={"from": "2020-01-01", "to": "2020-01-31"},
    )
    assert resp.status_code == 200
    cards = {c["metric"]: c for c in resp.json()["cards"]}
    assert cards["revenue"]["value"] == 0


# ── coverage ───────────────────────────────────────────────────────────────


async def test_data_coverage_reports_real_extents(client, seeded):
    resp = await client.get("/api/v1/data-coverage", headers=auth(seeded))
    assert resp.status_code == 200
    body = resp.json()

    assert body["sales"]["first_date"] == "2026-06-10"
    assert body["sales"]["last_date"] == "2026-06-12"
    assert body["sales"]["row_count"] == 4
    assert body["expenses"]["first_date"] == "2026-06-10"
    assert body["first_date"] == "2026-06-10"
    assert body["last_date"] == "2026-06-12"
    assert body["timezone"] == "Asia/Kathmandu"
    # uploads just happened, so every fact table carries an ingestion stamp
    assert body["sales"]["last_ingested_at"] is not None
    assert body["days_behind"] >= 0


async def test_coverage_requires_authentication(client):
    resp = await client.get("/api/v1/data-coverage")
    assert resp.status_code == 401


async def test_ingested_at_is_set_and_differs_from_business_date(client, seeded):
    """Upload date is recorded separately from the transaction date.

    The CSV is backdated to June; the ingestion stamp must be the upload
    moment, which is what "what did we load today?" relies on.
    """
    resp = await client.get("/api/v1/data-coverage", headers=auth(seeded))
    body = resp.json()
    ingested_day = body["sales"]["last_ingested_at"][:10]
    assert ingested_day == body["today"]
    assert ingested_day != body["sales"]["last_date"]
