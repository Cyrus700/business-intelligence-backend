"""Generate the FULL Nepali-SME retail dataset: 4 years of real, statistically
varied data — from the current day back to 4 years ago — covering every
analysis window (day → week → month → year → 4y) with huge volume and *all*
realistic variations:

  * mean variation: each day's order count is drawn from a Gaussian whose MEAN
    itself moves (growth trend + weekday + seasonal + festival multipliers) and
    whose STD is a fixed fraction of the mean — so the data has a real,
    explainable mean and spread, not flat noise.
  * huge multiple data: ~5-8x the demo volume; every dimension is exercised.
  * all other possible:
      - 7 weekdays (Nepali retail pattern: Sat busy, Sun quiet)
      - 3 segments (retail / wholesale / online)
      - 3 channels (store / distributor / online)
      - 7 provinces (Bagmati, Gandaki, Koshi, Madhesh, Lumbini, Karnali,
        Sudurpashchim) + extra cities
      - 7 categories incl. Festival items that spike at Dashain/Tihar
      - 4 years of Dashain + Tihar windows
      - injected revenue / expense anomalies (spikes & drops) every year
      - mild inflation on costs/prices across the years

Deterministic (seed 42). Output to seeds/output/:
  full_sales.csv, full_expenses.csv, full_inventory.csv

Usage:
    uv run python seeds/generate_full_data.py
"""

import json
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

YEARS_BACK = int(os.environ.get("SEED_YEARS_BACK", "4"))
VOLUME_MULT = float(os.environ.get("SEED_VOLUME_MULT", "6.0"))  # "huge multiple"
MEAN_VARIATION = float(os.environ.get("SEED_MEAN_VARIATION", "0.18"))  # std / mean

OUT = Path(__file__).parent / "output"
TODAY = date.today()
START = TODAY - timedelta(days=int(YEARS_BACK * 365.25))
END = TODAY

# ---------------------------------------------------------------------------
# Catalogue — expanded with more SKUs per category for "all possible" breadth.
# (sku, name, category, unit_cost, unit_price, popularity_weight)
# ---------------------------------------------------------------------------
PRODUCTS = [
    ("BEV-001", "Everest Tea 500g", "Beverages", 220, 320, 9),
    ("BEV-002", "Himal Coffee 200g", "Beverages", 380, 560, 5),
    ("BEV-003", "Mineral Water 1L (12pk)", "Beverages", 140, 220, 10),
    ("BEV-004", "Cola 2L (6pk)", "Beverages", 180, 280, 7),
    ("BEV-005", "Juice Mix 1L", "Beverages", 150, 240, 6),
    ("SNK-001", "Wai Wai Noodles (30pk)", "Snacks", 480, 640, 10),
    ("SNK-002", "Khaja Mix 400g", "Snacks", 160, 260, 7),
    ("SNK-003", "Biscuit Assorted Box", "Snacks", 210, 330, 6),
    ("SNK-004", "Chocolate Bar (24pk)", "Snacks", 300, 470, 5),
    ("SNK-005", "Potato Chips 150g", "Snacks", 90, 160, 8),
    ("DRY-001", "Basmati Rice 25kg", "Staples", 2800, 3600, 8),
    ("DRY-002", "Musuro Dal 5kg", "Staples", 700, 950, 7),
    ("DRY-003", "Mustard Oil 5L", "Staples", 1150, 1500, 6),
    ("DRY-004", "Chiura 5kg", "Staples", 380, 520, 5),
    ("DRY-005", "Pulao Masala 200g", "Staples", 120, 220, 4),
    ("HHD-001", "Detergent Powder 3kg", "Household", 420, 620, 6),
    ("HHD-002", "Dish Soap (6pk)", "Household", 180, 290, 5),
    ("HHD-003", "LED Bulb 9W (4pk)", "Household", 360, 560, 4),
    ("HHD-004", "Garbage Bags (100pk)", "Household", 140, 240, 4),
    ("PCR-001", "Herbal Soap (12pk)", "Personal Care", 300, 460, 5),
    ("PCR-002", "Shampoo 650ml", "Personal Care", 340, 520, 4),
    ("PCR-003", "Toothpaste Family (6pk)", "Personal Care", 390, 580, 4),
    ("PCR-004", "Sanitary Pack (10pk)", "Personal Care", 220, 360, 3),
    ("ELC-001", "Rice Cooker 1.8L", "Electronics", 2600, 3900, 2),
    ("ELC-002", "Electric Kettle 2L", "Electronics", 1500, 2350, 2),
    ("ELC-003", "Ceiling Fan 56in", "Electronics", 2900, 4300, 1),
    ("ELC-004", "Smartphone 128GB", "Electronics", 18000, 25000, 1),
    ("FES-001", "Diyo & Batti Set", "Festival", 90, 180, 2),
    ("FES-002", "Sel Roti Mix 1kg", "Festival", 170, 280, 2),
    ("FES-003", "Marigold Garland (10pk)", "Festival", 150, 300, 2),
    ("FES-004", "Gift Hamper Deluxe", "Festival", 1400, 2200, 1),
    ("FES-005", "Sparkler Pack", "Festival", 260, 420, 1),
]

# (name, segment, city, region)
CUSTOMERS = [
    ("Bhatbhateni Retail KTM", "wholesale", "Kathmandu", "Bagmati"),
    ("Salesberry Lalitpur", "wholesale", "Lalitpur", "Bagmati"),
    ("Namaste Mart", "retail", "Kathmandu", "Bagmati"),
    ("Gurung Kirana Pasal", "retail", "Pokhara", "Gandaki"),
    ("Lakeside Store", "retail", "Pokhara", "Gandaki"),
    ("Machhapuchhre Suppliers", "wholesale", "Pokhara", "Gandaki"),
    ("Everest Traders", "wholesale", "Biratnagar", "Koshi"),
    ("Koshi Retail House", "retail", "Biratnagar", "Koshi"),
    ("Janaki Store", "retail", "Janakpur", "Madhesh"),
    ("Terai Wholesale Hub", "wholesale", "Birgunj", "Madhesh"),
    ("Lumbini Mart", "retail", "Butwal", "Lumbini"),
    ("Siddhartha Suppliers", "wholesale", "Bhairahawa", "Lumbini"),
    ("Daraz Online Nepal", "online", "Kathmandu", "Bagmati"),
    ("SastoDeal Online", "online", "Kathmandu", "Bagmati"),
    ("Himalayan e-Shop", "online", "Pokhara", "Gandaki"),
    ("Chitwan Kirana Center", "retail", "Bharatpur", "Bagmati"),
    ("Narayani Traders", "wholesale", "Bharatpur", "Bagmati"),
    ("Dharan Retail Corner", "retail", "Dharan", "Koshi"),
    ("Mechi Suppliers", "wholesale", "Birtamod", "Koshi"),
    ("Karnali Store", "retail", "Surkhet", "Karnali"),
    ("Farwest Mart", "retail", "Dhangadhi", "Sudurpashchim"),
    ("Seti Wholesale", "wholesale", "Mahendranagar", "Sudurpashchim"),
    ("Gandaki Retail", "retail", "Baglung", "Gandaki"),
    ("Terai Retail", "retail", "Rajbiraj", "Madhesh"),
]
CHANNELS = {"retail": "store", "wholesale": "distributor", "online": "online"}

# Dashain + Tihar windows for every covered year (pre-festival ramp handled in code).
FESTIVALS = [
    (date(2022, 10, 1), date(2022, 10, 10), "dashain"),
    (date(2022, 10, 22), date(2022, 10, 26), "tihar"),
    (date(2023, 10, 15), date(2023, 10, 24), "dashain"),
    (date(2023, 11, 10), date(2023, 11, 15), "tihar"),
    (date(2024, 10, 3), date(2024, 10, 12), "dashain"),
    (date(2024, 10, 29), date(2024, 11, 3), "tihar"),
    (date(2025, 9, 22), date(2025, 10, 2), "dashain"),
    (date(2025, 10, 18), date(2025, 10, 23), "tihar"),
    (date(2026, 10, 11), date(2026, 10, 20), "dashain"),
    (date(2026, 11, 6), date(2026, 11, 11), "tihar"),
]


def festival_boost(d: date) -> tuple[float, bool]:
    for start, end, _ in FESTIVALS:
        if start <= d <= end:
            return 2.1, True
        if start - timedelta(days=10) <= d < start:  # pre-festival stocking ramp
            return 1.4, True
    return 1.0, False


# Seasonal demand (Nepal: monsoon dip Jun-Aug, winter bump Nov-Jan, festival Q4).
SEASON_MULT = {
    1: 1.10, 2: 1.05, 3: 1.0, 4: 1.0, 5: 0.95, 6: 0.88,
    7: 0.85, 8: 0.9, 9: 1.05, 10: 1.25, 11: 1.30, 12: 1.20,
}

# Injected ground-truth anomalies spread evenly across all 4 years.
ANOMALIES = [
    {"date": "2022-09-14", "metric": "revenue", "kind": "spike", "factor": 3.2},
    {"date": "2022-12-05", "metric": "revenue", "kind": "drop", "factor": 0.15},
    {"date": "2023-01-21", "metric": "revenue", "kind": "spike", "factor": 2.8},
    {"date": "2023-03-08", "metric": "expense_total", "kind": "spike", "factor": 4.0},
    {"date": "2023-05-30", "metric": "revenue", "kind": "drop", "factor": 0.2},
    {"date": "2023-07-19", "metric": "revenue", "kind": "spike", "factor": 3.5},
    {"date": "2023-09-02", "metric": "expense_total", "kind": "spike", "factor": 3.5},
    {"date": "2023-12-14", "metric": "revenue", "kind": "drop", "factor": 0.18},
    {"date": "2024-02-11", "metric": "revenue", "kind": "spike", "factor": 3.0},
    {"date": "2024-04-25", "metric": "expense_total", "kind": "spike", "factor": 4.5},
    {"date": "2024-06-07", "metric": "revenue", "kind": "drop", "factor": 0.22},
    {"date": "2024-08-16", "metric": "revenue", "kind": "spike", "factor": 2.9},
    {"date": "2024-11-29", "metric": "revenue", "kind": "drop", "factor": 0.2},
    {"date": "2025-01-09", "metric": "revenue", "kind": "spike", "factor": 3.4},
    {"date": "2025-02-27", "metric": "expense_total", "kind": "spike", "factor": 3.8},
    {"date": "2025-04-17", "metric": "revenue", "kind": "drop", "factor": 0.17},
    {"date": "2025-05-23", "metric": "revenue", "kind": "spike", "factor": 3.1},
    {"date": "2025-06-19", "metric": "revenue", "kind": "spike", "factor": 2.7},
    {"date": "2025-07-14", "metric": "revenue", "kind": "spike", "factor": 2.9},
    {"date": "2025-07-28", "metric": "expense_total", "kind": "spike", "factor": 3.6},
    {"date": "2026-01-09", "metric": "revenue", "kind": "spike", "factor": 3.4},
    {"date": "2026-02-27", "metric": "expense_total", "kind": "spike", "factor": 3.8},
    {"date": "2026-04-17", "metric": "revenue", "kind": "drop", "factor": 0.17},
    {"date": "2026-05-23", "metric": "revenue", "kind": "spike", "factor": 3.1},
    {"date": "2026-06-19", "metric": "revenue", "kind": "spike", "factor": 2.7},
    {"date": "2026-07-14", "metric": "revenue", "kind": "spike", "factor": 2.9},
    {"date": "2026-07-28", "metric": "expense_total", "kind": "spike", "factor": 3.6},
]
ANOMALY_BY_DATE = {a["date"]: a for a in ANOMALIES}


def inflation_factor(d: date) -> float:
    """Mild ~6%/yr cost & price inflation across the window."""
    years_in = (d - START).days / 365.25
    return 1.0 + 0.06 * years_in


def gen_sales() -> pd.DataFrame:
    rows = []
    n_days = (END - START).days + 1
    for i in range(n_days):
        d = START + timedelta(days=i)
        years_in = i / 365.25

        # ---- MEAN: a moving daily order mean (growth + weekday + season + fest) ----
        base_mean = 26 * VOLUME_MULT * (1 + 0.18 * years_in)
        weekday_mult = [1.0, 0.95, 0.95, 1.0, 1.15, 1.3, 0.7][d.weekday()]
        fest_mult, in_festival = festival_boost(d)
        mean_orders = base_mean * weekday_mult * SEASON_MULT[d.month] * fest_mult

        # ---- VARIATION: draw the actual count from a Gaussian around that mean ----
        n_orders = max(3, int(RNG.normal(mean_orders, mean_orders * MEAN_VARIATION)))

        anomaly = ANOMALY_BY_DATE.get(d.isoformat())
        if anomaly and anomaly["metric"] == "revenue":
            n_orders = max(2, int(n_orders * anomaly["factor"]))

        weights = np.array(
            [p[5] * (3.0 if in_festival and p[0].startswith("FES") else 1.0) for p in PRODUCTS],
            dtype=float,
        )
        weights /= weights.sum()
        infl = inflation_factor(d)
        for _ in range(n_orders):
            sku, name, cat, cost, price, _w = PRODUCTS[RNG.choice(len(PRODUCTS), p=weights)]
            cust, seg, city, region = CUSTOMERS[RNG.integers(len(CUSTOMERS))]
            qty = int(max(1, RNG.poisson(6 if seg == "wholesale" else 2)))
            unit_price = round(price * infl, 2)
            discount = round(unit_price * qty * float(RNG.choice([0, 0, 0, 0.05, 0.1])), 2)
            rows.append(
                {
                    "date": d.isoformat(),
                    "sku": sku,
                    "product_name": name,
                    "category": cat,
                    "customer": cust,
                    "segment": seg,
                    "city": city,
                    "region": region,
                    "channel": CHANNELS[seg],
                    "quantity": qty,
                    "unit_price": unit_price,
                    "discount": discount,
                }
            )
    return pd.DataFrame(rows)


def gen_expenses() -> pd.DataFrame:
    rows = []
    d = START
    while d <= END:
        infl = inflation_factor(d)
        if d.day == 1:
            rows.append(
                {
                    "date": d.isoformat(),
                    "category": "rent",
                    "amount": round(185000 * infl, 2),
                    "department": "operations",
                    "description": "Warehouse & office rent",
                }
            )
        if d.day == 25:
            rows.append(
                {
                    "date": d.isoformat(),
                    "category": "salaries",
                    "amount": round(520000 * (1 + 0.08 * ((d.year - START.year) + d.month / 12)) * infl, 2),
                    "department": "hr",
                    "description": "Monthly payroll",
                }
            )
        if d.day == 10:
            rows.append(
                {
                    "date": d.isoformat(),
                    "category": "utilities",
                    "amount": round(float(RNG.normal(38000, 6000)) * infl, 2),
                    "department": "operations",
                    "description": "Electricity, internet, water",
                }
            )
        if d.weekday() == 0 and RNG.random() < 0.7:
            rows.append(
                {
                    "date": d.isoformat(),
                    "category": "marketing",
                    "amount": round(float(RNG.normal(24000, 9000)) * infl, 2),
                    "department": "sales",
                    "description": "Weekly promotions",
                }
            )
        if RNG.random() < 0.85:
            rows.append(
                {
                    "date": d.isoformat(),
                    "category": "logistics",
                    "amount": round(float(RNG.normal(9500, 3200)) * infl, 2),
                    "department": "operations",
                    "description": "Delivery & fuel",
                }
            )

        anomaly = ANOMALY_BY_DATE.get(d.isoformat())
        if anomaly and anomaly["metric"] == "expense_total":
            rows.append(
                {
                    "date": d.isoformat(),
                    "category": "other",
                    "amount": round(60000 * anomaly["factor"] * infl, 2),
                    "department": "operations",
                    "description": "Unplanned equipment repair",
                }
            )
        d += timedelta(days=1)
    frame = pd.DataFrame(rows)
    frame["amount"] = frame["amount"].clip(lower=500)
    return frame


def gen_inventory(sales: pd.DataFrame) -> pd.DataFrame:
    sales = sales.copy()
    sales["month"] = pd.to_datetime(sales["date"]).dt.to_period("M")
    monthly_qty = sales.groupby(["month", "sku"])["quantity"].sum()
    rows = []
    stock = {p[0]: int(RNG.integers(400, 900)) for p in PRODUCTS}
    for month in sorted(sales["month"].unique()):
        snap = (month.to_timestamp() + pd.offsets.MonthEnd(0)).date()
        if snap > END:
            snap = END
        for sku, *_ in PRODUCTS:
            sold = int(monthly_qty.get((month, sku), 0))
            restock = int(sold * float(RNG.uniform(0.85, 1.2)))
            stock[sku] = max(0, stock[sku] - sold + restock)
            rows.append(
                {
                    "date": snap.isoformat(),
                    "sku": sku,
                    "quantity_on_hand": stock[sku],
                    "reorder_level": 120,
                    "warehouse": "main",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sales = gen_sales()
    expenses = gen_expenses()
    inventory = gen_inventory(sales)

    sales.to_csv(OUT / "full_sales.csv", index=False)
    expenses.to_csv(OUT / "full_expenses.csv", index=False)
    inventory.to_csv(OUT / "full_inventory.csv", index=False)

    (Path(__file__).parent / "injected_anomalies.json").write_text(json.dumps(ANOMALIES, indent=2))

    revenue = (sales["quantity"] * sales["unit_price"] - sales["discount"]).sum()
    print(f"window:          {START} -> {END}  ({YEARS_BACK} years)")
    print(f"sales rows:      {len(sales):>9,}")
    print(f"expense rows:    {len(expenses):>9,}")
    print(f"inventory rows:  {len(inventory):>9,}")
    print(f"4y gross revenue: NPR {revenue:,.0f}")
    print(f"injected anomalies: {len(ANOMALIES)}")
    print(f"volume_mult={VOLUME_MULT} mean_variation={MEAN_VARIATION}")
    # Quick window sanity
    for label, mask in [
        ("today", sales["date"] == TODAY.isoformat()),
        ("this_week", pd.to_datetime(sales["date"]).dt.isocalendar().week == TODAY.isocalendar()[1]),
        ("this_month", sales["date"].str.startswith(TODAY.strftime("%Y-%m"))),
        ("this_year", sales["date"].str.startswith(str(TODAY.year))),
        ("year_4_ago", sales["date"].str.startswith(str(START.year))),
    ]:
        print(f"  rows {label:<10}: {int(mask.sum()):>9,}")


if __name__ == "__main__":
    main()
