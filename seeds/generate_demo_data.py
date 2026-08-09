"""Generate the synthetic Nepali-SME retail dataset (see docs/05-ml-plan.md).

Deterministic (seed 42). Produces into seeds/output/:
  sales.csv, expenses.csv, inventory.csv        — 3-year baseline
  sales_2026_06.csv                             — viva demo: clean incremental month
  sales_with_errors.csv                         — viva demo: file with bad rows
  injected_anomalies.json                       — ground-truth labels for Phase 4 evaluation

Usage: uv run python seeds/generate_demo_data.py
"""

import json
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
random.seed(42)

OUT = Path(__file__).parent / "output"
START = date(2023, 7, 1)
END = date.today()  # generate through today so every dashboard window has data

PRODUCTS = [
    # sku, name, category, unit_cost, unit_price, popularity weight
    ("BEV-001", "Everest Tea 500g", "Beverages", 220, 320, 9),
    ("BEV-002", "Himal Coffee 200g", "Beverages", 380, 560, 5),
    ("BEV-003", "Mineral Water 1L (12pk)", "Beverages", 140, 220, 10),
    ("SNK-001", "Wai Wai Noodles (30pk)", "Snacks", 480, 640, 10),
    ("SNK-002", "Khaja Mix 400g", "Snacks", 160, 260, 7),
    ("SNK-003", "Biscuit Assorted Box", "Snacks", 210, 330, 6),
    ("DRY-001", "Basmati Rice 25kg", "Staples", 2800, 3600, 8),
    ("DRY-002", "Musuro Dal 5kg", "Staples", 700, 950, 7),
    ("DRY-003", "Mustard Oil 5L", "Staples", 1150, 1500, 6),
    ("DRY-004", "Chiura 5kg", "Staples", 380, 520, 5),
    ("HHD-001", "Detergent Powder 3kg", "Household", 420, 620, 6),
    ("HHD-002", "Dish Soap (6pk)", "Household", 180, 290, 5),
    ("HHD-003", "LED Bulb 9W (4pk)", "Household", 360, 560, 4),
    ("PCR-001", "Herbal Soap (12pk)", "Personal Care", 300, 460, 5),
    ("PCR-002", "Shampoo 650ml", "Personal Care", 340, 520, 4),
    ("PCR-003", "Toothpaste Family (6pk)", "Personal Care", 390, 580, 4),
    ("ELC-001", "Rice Cooker 1.8L", "Electronics", 2600, 3900, 2),
    ("ELC-002", "Electric Kettle 2L", "Electronics", 1500, 2350, 2),
    ("ELC-003", "Ceiling Fan 56in", "Electronics", 2900, 4300, 1),
    ("FES-001", "Diyo & Batti Set", "Festival", 90, 180, 2),  # spikes at Tihar
    ("FES-002", "Sel Roti Mix 1kg", "Festival", 170, 280, 2),  # spikes at Dashain
    ("FES-003", "Marigold Garland (10pk)", "Festival", 150, 300, 2),
    ("FES-004", "Gift Hamper Deluxe", "Festival", 1400, 2200, 1),
    ("FES-005", "Sparkler Pack", "Festival", 260, 420, 1),
]

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
]

CHANNELS = {"retail": "store", "wholesale": "distributor", "online": "online"}

# Approximate Gregorian windows for the big festival demand seasons
FESTIVALS = [
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


# ---- injected ground-truth anomalies (Phase 4 evaluation labels) ----
ANOMALIES = [
    {"date": "2023-09-14", "metric": "revenue", "kind": "spike", "factor": 3.2},
    {"date": "2023-12-05", "metric": "revenue", "kind": "drop", "factor": 0.15},
    {"date": "2024-01-21", "metric": "revenue", "kind": "spike", "factor": 2.8},
    {"date": "2024-03-08", "metric": "expense_total", "kind": "spike", "factor": 4.0},
    {"date": "2024-05-30", "metric": "revenue", "kind": "drop", "factor": 0.2},
    {"date": "2024-07-19", "metric": "revenue", "kind": "spike", "factor": 3.5},
    {"date": "2024-09-02", "metric": "expense_total", "kind": "spike", "factor": 3.5},
    {"date": "2024-12-14", "metric": "revenue", "kind": "drop", "factor": 0.18},
    {"date": "2025-02-11", "metric": "revenue", "kind": "spike", "factor": 3.0},
    {"date": "2025-04-25", "metric": "expense_total", "kind": "spike", "factor": 4.5},
    {"date": "2025-06-07", "metric": "revenue", "kind": "drop", "factor": 0.22},
    {"date": "2025-08-16", "metric": "revenue", "kind": "spike", "factor": 2.9},
    {"date": "2025-11-29", "metric": "revenue", "kind": "drop", "factor": 0.2},
    {"date": "2026-01-09", "metric": "revenue", "kind": "spike", "factor": 3.4},
    {"date": "2026-02-27", "metric": "expense_total", "kind": "spike", "factor": 3.8},
    {"date": "2026-04-17", "metric": "revenue", "kind": "drop", "factor": 0.17},
    {"date": "2026-05-23", "metric": "revenue", "kind": "spike", "factor": 3.1},
    {"date": "2026-06-19", "metric": "revenue", "kind": "spike", "factor": 2.7},
    {"date": "2026-07-14", "metric": "revenue", "kind": "spike", "factor": 2.9},
    {"date": "2026-07-28", "metric": "expense_total", "kind": "spike", "factor": 3.6},
]
ANOMALY_BY_DATE = {a["date"]: a for a in ANOMALIES}


def gen_sales() -> pd.DataFrame:
    rows = []
    n_days = (END - START).days + 1
    for i in range(n_days):
        d = START + timedelta(days=i)
        years_in = i / 365.25
        base = 26 * (1 + 0.18 * years_in)  # gentle growth trend
        weekday_mult = [1.0, 0.95, 0.95, 1.0, 1.15, 1.3, 0.7][
            d.weekday()
        ]  # Sat=5 busy, Sun=6 (Nepal wk)
        fest_mult, in_festival = festival_boost(d)
        n_orders = max(3, int(rng.normal(base * weekday_mult * fest_mult, base * 0.14)))

        anomaly = ANOMALY_BY_DATE.get(d.isoformat())
        if anomaly and anomaly["metric"] == "revenue":
            n_orders = max(2, int(n_orders * anomaly["factor"]))

        weights = np.array(
            [p[5] * (3.0 if in_festival and p[0].startswith("FES") else 1.0) for p in PRODUCTS],
            dtype=float,
        )
        weights /= weights.sum()
        for _ in range(n_orders):
            sku, name, cat, cost, price, _w = PRODUCTS[rng.choice(len(PRODUCTS), p=weights)]
            cust, seg, city, region = CUSTOMERS[rng.integers(len(CUSTOMERS))]
            qty = int(max(1, rng.poisson(6 if seg == "wholesale" else 2)))
            discount = round(float(price * qty) * float(rng.choice([0, 0, 0, 0.05, 0.1])), 2)
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
                    "unit_price": price,
                    "discount": discount,
                }
            )
    return pd.DataFrame(rows)


def gen_expenses() -> pd.DataFrame:
    rows = []
    d = START
    while d <= END:
        if d.day == 1:
            rows.append(
                {
                    "date": d.isoformat(),
                    "category": "rent",
                    "amount": 185000,
                    "department": "operations",
                    "description": "Warehouse & office rent",
                }
            )
        if d.day == 25:
            rows.append(
                {
                    "date": d.isoformat(),
                    "category": "salaries",
                    "amount": round(520000 * (1 + 0.08 * ((d.year - 2023) + d.month / 12)), 2),
                    "department": "hr",
                    "description": "Monthly payroll",
                }
            )
        if d.day == 10:
            rows.append(
                {
                    "date": d.isoformat(),
                    "category": "utilities",
                    "amount": round(float(rng.normal(38000, 6000)), 2),
                    "department": "operations",
                    "description": "Electricity, internet, water",
                }
            )
        if d.weekday() == 0 and rng.random() < 0.7:
            rows.append(
                {
                    "date": d.isoformat(),
                    "category": "marketing",
                    "amount": round(float(rng.normal(24000, 9000)), 2),
                    "department": "sales",
                    "description": "Weekly promotions",
                }
            )
        if rng.random() < 0.85:
            rows.append(
                {
                    "date": d.isoformat(),
                    "category": "logistics",
                    "amount": round(float(rng.normal(9500, 3200)), 2),
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
                    "amount": round(60000 * anomaly["factor"], 2),
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
    stock = {p[0]: int(rng.integers(400, 900)) for p in PRODUCTS}
    for month in sorted(sales["month"].unique()):
        snap = (month.to_timestamp() + pd.offsets.MonthEnd(0)).date()
        if snap > END:
            snap = END
        for sku, *_ in PRODUCTS:
            sold = int(monthly_qty.get((month, sku), 0))
            restock = int(sold * float(rng.uniform(0.85, 1.2)))
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


def gen_error_file(sales: pd.DataFrame) -> pd.DataFrame:
    good = sales.tail(14).copy().reset_index(drop=True)
    bad = pd.DataFrame(
        [
            {"date": "not-a-date", "sku": "BEV-001", "quantity": 2, "unit_price": 320},
            {"date": "2026-06-15", "sku": "", "quantity": 2, "unit_price": 320},
            {"date": "2026-06-15", "sku": "BEV-001", "quantity": -4, "unit_price": 320},
            {"date": "2026-06-15", "sku": "BEV-001", "quantity": 2, "unit_price": "free"},
            {"date": "2099-01-01", "sku": "BEV-001", "quantity": 2, "unit_price": 320},
            {
                "date": "2026-06-15",
                "sku": "BEV-001",
                "quantity": 1,
                "unit_price": 100,
                "discount": 500,
            },
        ]
    )
    return pd.concat([good, bad], ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sales = gen_sales()
    expenses = gen_expenses()
    inventory = gen_inventory(sales)

    sales.to_csv(OUT / "sales.csv", index=False)
    expenses.to_csv(OUT / "expenses.csv", index=False)
    inventory.to_csv(OUT / "inventory.csv", index=False)

    june = sales[sales["date"].str.startswith("2026-06")]
    june.to_csv(OUT / "sales_2026_06.csv", index=False)
    gen_error_file(sales).to_csv(OUT / "sales_with_errors.csv", index=False)

    (Path(__file__).parent / "injected_anomalies.json").write_text(json.dumps(ANOMALIES, indent=2))

    revenue = (sales["quantity"] * sales["unit_price"] - sales["discount"]).sum()
    print(f"sales rows:     {len(sales):>7,}")
    print(f"expense rows:   {len(expenses):>7,}")
    print(f"inventory rows: {len(inventory):>7,}")
    print(f"3y gross revenue: NPR {revenue:,.0f}")
    print(f"injected anomalies: {len(ANOMALIES)}")


if __name__ == "__main__":
    main()
