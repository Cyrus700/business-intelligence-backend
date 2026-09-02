"""Seed 2 isolated businesses with real demo data + 3 roles each.

Businesses:
  1) Himalayan Traders Pvt Ltd  (slug: himalayan-traders)
  2) Everest Retail House        (slug: everest-retail)

Each business gets:
  - 1 admin, 1 manager, 1 analyst  (total 6 accounts)
  - Its own products (org-scoped SKU uniqueness), 3 years of sales/expenses/inventory
    generated deterministically so analytics dashboards are immediately useful.
  - Per-org kpi_snapshots rebuilt.

Super-admin (platform operator) is kept from ADMIN_EMAIL if set, otherwise the
legacy admin is left untouched.

Idempotent — safe to run multiple times (skips existing orgs/users, regenerates
only if --force or if business has < 100 sales rows).

Usage:
  uv run python seeds/seed_two_businesses.py
  uv run python seeds/seed_two_businesses.py --force

Credentials printed at end; also usable via UI Register Business / Invite.
"""
import asyncio
import sys
import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path
import uuid

import bcrypt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select, text, func
from app.core.database import get_session_factory
from app.models import Organization, Profile, Product
from app.models.warehouse import SalesTransaction, Expense, InventoryLevel
from app.models.integration import DataSource, EtlJob
from app.services.analytics.kpi_builder import rebuild_kpi_snapshots

# --- Business definitions ---
BUSINESSES = [
    {
        "name": "Himalayan Traders Pvt Ltd",
        "slug": "himalayan-traders",
        "email_domain": "himalayan.example.com",
        "seed": 42,
        "base_orders": 26,  # slightly larger
        "regions": ["Kathmandu", "Pokhara", "Bhairahawa"],  # focus west/mid
    },
    {
        "name": "Everest Retail House",
        "slug": "everest-retail",
        "email_domain": "everest.example.com",
        "seed": 1337,
        "base_orders": 19,  # smaller
        "regions": ["Biratnagar", "Janakpur", "Dharan"],  # east focus
    },
]

# Roles per business
ROLES = [
    ("admin", "Admin@123456", "Business Admin"),
    ("manager", "Manager@123456", "Operations Manager"),
    ("analyst", "Analyst@123456", "Business Analyst"),
]

# Shared catalog (per-org, so same SKUs can repeat across orgs)
PRODUCTS = [
    ("BEV-001", "Everest Tea 500g", "Beverages", 220, 320),
    ("BEV-002", "Himal Coffee 200g", "Beverages", 380, 560),
    ("BEV-003", "Mineral Water 1L (12pk)", "Beverages", 140, 220),
    ("SNK-001", "Wai Wai Noodles (30pk)", "Snacks", 480, 640),
    ("SNK-002", "Khaja Mix 400g", "Snacks", 160, 260),
    ("SNK-003", "Biscuit Assorted Box", "Snacks", 210, 330),
    ("DRY-001", "Basmati Rice 25kg", "Staples", 2800, 3600),
    ("DRY-002", "Musuro Dal 5kg", "Staples", 700, 950),
    ("DRY-003", "Mustard Oil 5L", "Staples", 1150, 1500),
    ("HHD-001", "Detergent Powder 3kg", "Household", 420, 620),
    ("PCR-001", "Herbal Soap (12pk)", "Personal Care", 300, 460),
    ("ELC-001", "Rice Cooker 1.8L", "Electronics", 2600, 3900),
]

CUSTOMERS = [
    ("Bhatbhateni Retail KTM", "wholesale", "Kathmandu", "Bagmati"),
    ("Gurung Kirana Pasal", "retail", "Pokhara", "Gandaki"),
    ("Everest Traders", "wholesale", "Biratnagar", "Koshi"),
    ("Janaki Store", "retail", "Janakpur", "Madhesh"),
    ("Daraz Online Nepal", "online", "Kathmandu", "Bagmati"),
    ("Lumbini Mart", "retail", "Butwal", "Lumbini"),
]
CHANNELS = {"retail": "store", "wholesale": "distributor", "online": "online"}
FESTIVALS = [
    (date(2023, 10, 15), date(2023, 10, 24)),
    (date(2023, 11, 10), date(2023, 11, 15)),
    (date(2024, 10, 3), date(2024, 10, 12)),
    (date(2024, 10, 29), date(2024, 11, 3)),
    (date(2025, 9, 22), date(2025, 10, 2)),
    (date(2025, 10, 18), date(2025, 10, 23)),
    (date(2026, 10, 11), date(2026, 10, 20)),
]

def festival_boost(d: date) -> float:
    for s, e in FESTIVALS:
        if s <= d <= e:
            return 2.1
        if s - timedelta(days=10) <= d < s:
            return 1.4
    return 1.0

ANOMALIES = [
    {"date": "2024-03-08", "factor": 4.0},
    {"date": "2025-04-25", "factor": 4.5},
]
ANOMALY_BY_DATE = {a["date"]: a["factor"] for a in ANOMALIES}

START = date(2023, 7, 1)
END = date.today()

def _hash(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()

async def ensure_org(name: str, slug: str) -> Organization:
    async with get_session_factory()() as s:
        org = (await s.execute(select(Organization).where(Organization.slug == slug))).scalar_one_or_none()
        if org:
            print(f"Org exists: {org.name} ({org.slug}) id={org.id}")
            return org
        # also check by name case-insensitive
        org2 = (await s.execute(select(Organization).where(func.lower(Organization.name) == name.lower()))).scalar_one_or_none()
        if org2:
            print(f"Org exists by name: {org2.name} id={org2.id}")
            return org2
        org = Organization(name=name, slug=slug, is_legacy=False)
        s.add(org)
        await s.commit()
        await s.refresh(org)
        print(f"Created org: {org.name} ({org.slug}) id={org.id}")
        return org

async def ensure_user(email: str, password: str, role: str, full_name: str, org_id, department="General") -> Profile:
    async with get_session_factory()() as s:
        existing = (await s.execute(select(Profile).where(func.lower(Profile.email) == email.lower()))).scalar_one_or_none()
        if existing:
            # update if org/role drift
            changed = False
            if existing.org_id != org_id:
                existing.org_id = org_id
                changed = True
            if existing.role != role:
                existing.role = role
                changed = True
            if existing.is_active is False:
                existing.is_active = True
                changed = True
            if changed:
                await s.commit()
                print(f"Updated {email} -> org={org_id} role={role}")
            else:
                print(f"User exists: {email} ({role}) org={org_id}")
            return existing
        uid = uuid.uuid5(uuid.NAMESPACE_URL, f"https://seed/{email.lower()}")
        # check id collision (email case change previously) - generate random if collides
        exists_id = await s.get(Profile, uid)
        if exists_id:
            uid = uuid.uuid4()
        profile = Profile(
            id=uid,
            email=email.lower(),
            password_hash=_hash(password),
            full_name=full_name,
            role=role,
            department=department,
            org_id=org_id,
            is_active=True,
            is_super_admin=False,
        )
        s.add(profile)
        await s.commit()
        print(f"Seeded {email} ({role}) org={org_id}")
        return profile

async def ensure_products(org_id) -> dict[str, uuid.UUID]:
    async with get_session_factory()() as s:
        existing = {p.sku: p.id for p in (await s.execute(select(Product).where(Product.org_id == org_id))).scalars().all()}
        wanted = set(sku for sku, *_ in PRODUCTS)
        missing = wanted - set(existing.keys())
        for sku, name, cat, cost, price in PRODUCTS:
            if sku in missing:
                p = Product(sku=sku, name=name, category=cat, unit_cost=cost, unit_price=price, org_id=org_id)
                s.add(p)
        if missing:
            await s.commit()
            existing = {p.sku: p.id for p in (await s.execute(select(Product).where(Product.org_id == org_id))).scalars().all()}
            print(f"Created {len(missing)} products for org {org_id}")
        else:
            print(f"Products already exist for org {org_id}: {len(existing)}")
        return existing

async def count_sales(org_id) -> int:
    async with get_session_factory()() as s:
        return (await s.execute(select(func.count()).select_from(SalesTransaction).where(SalesTransaction.org_id == org_id))).scalar_one()

async def seed_business_data(business: dict, org_id, force: bool = False):
    cnt = await count_sales(org_id)
    if cnt > 100 and not force:
        print(f"Business {business['name']} already has {cnt} sales rows — skipping generation (use --force to regenerate)")
        return
    if force and cnt > 0:
        # truncate only this org's data for regeneration (order matters due FK)
        async with get_session_factory()() as s:
            await s.execute(text("DELETE FROM kpi_snapshots WHERE org_id = :oid"), {"oid": str(org_id)})
            await s.execute(text("DELETE FROM forecasts WHERE org_id = :oid"), {"oid": str(org_id)})
            await s.execute(text("DELETE FROM ml_models WHERE org_id = :oid"), {"oid": str(org_id)})
            await s.execute(text("DELETE FROM anomalies WHERE org_id = :oid"), {"oid": str(org_id)})
            await s.execute(text("DELETE FROM insights WHERE org_id = :oid"), {"oid": str(org_id)})
            await s.execute(text("DELETE FROM sales_transactions WHERE org_id = :oid"), {"oid": str(org_id)})
            await s.execute(text("DELETE FROM expenses WHERE org_id = :oid"), {"oid": str(org_id)})
            await s.execute(text("DELETE FROM inventory_levels WHERE org_id = :oid"), {"oid": str(org_id)})
            await s.commit()
            print(f"Cleared existing data for {business['name']} (force)")

    rng = np.random.default_rng(business["seed"])
    random.seed(business["seed"])
    prod_map = await ensure_products(org_id)

    # Prepare customers per business regions
    biz_customers = [c for c in CUSTOMERS if c[3] in business["regions"]] or CUSTOMERS
    # ensure at least 4 customers
    if len(biz_customers) < 4:
        biz_customers = CUSTOMERS[:4]

    sales_rows = []
    expense_rows = []
    n_days = (END - START).days + 1
    for i in range(n_days):
        d = START + timedelta(days=i)
        years_in = i / 365.25
        base = business["base_orders"] * (1 + 0.18 * years_in)
        weekday_mult = [1.0, 0.95, 0.95, 1.0, 1.15, 1.3, 0.7][d.weekday()]
        fest_mult = festival_boost(d)
        n_orders = max(3, int(rng.normal(base * weekday_mult * fest_mult, base * 0.14)))
        factor = ANOMALY_BY_DATE.get(d.isoformat())
        if factor and random.random() < 0.7:
            n_orders = max(2, int(n_orders * factor * 0.6))

        weights = np.array([1.0]*len(PRODUCTS))
        weights /= weights.sum()
        for _ in range(n_orders):
            sku, name, cat, cost, price = PRODUCTS[int(rng.choice(len(PRODUCTS), p=weights))]
            cust, seg, city, region = biz_customers[int(rng.integers(len(biz_customers)))]
            qty = int(max(1, rng.poisson(6 if seg == "wholesale" else 2)))
            discount = round(float(price * qty) * float(rng.choice([0,0,0,0.05,0.1])),2)
            total = price * qty - discount
            # row_hash per org - must be globally unique per row, so include uuid nonce
            if business["slug"] == "everest-retail":
                total = round(total * 0.92, 2)  # slightly lower avg price for second business
            sales_rows.append({
                "txn_date": d,
                "product_id": prod_map[sku],
                "quantity": qty,
                "unit_price": price,
                "discount": discount,
                "total_amount": total,
                "channel": CHANNELS[seg],
                "region": region,
                "row_hash": f"{org_id.hex[:4]}-{uuid.uuid4().hex[:14]}-{d.isoformat()}",
                "ingested_at": date.today(),
            })
        # expenses: similar but lighter
        if d.day == 1:
            expense_rows.append({"expense_date": d, "category": "rent", "amount": 185000 if business["slug"]=="himalayan-traders" else 140000, "department": "operations"})
        if d.day == 10:
            expense_rows.append({"expense_date": d, "category": "utilities", "amount": max(500, round(float(rng.normal(38000,6000)),2)), "department": "operations"})
        if d.day == 25:
            amt = round(520000 * (1 + 0.08*((d.year-2023)+d.month/12)),2)
            if business["slug"]=="everest-retail":
                amt = round(amt*0.85,2)
            expense_rows.append({"expense_date": d, "category": "salaries", "amount": max(500, amt), "department": "hr"})
        if d.weekday()==0 and rng.random()<0.7:
            expense_rows.append({"expense_date": d, "category": "marketing", "amount": max(500, round(float(rng.normal(24000,9000)),2)), "department": "sales"})
        if rng.random()<0.85:
            expense_rows.append({"expense_date": d, "category": "logistics", "amount": max(500, round(float(rng.normal(9500,3200)),2)), "department": "operations"})

    # Bulk insert via ORM with org_id
    # Use chunks to avoid too large single commit
    async with get_session_factory()() as s:
        # sales
        for chunk_start in range(0, len(sales_rows), 2000):
            chunk = sales_rows[chunk_start: chunk_start+2000]
            for r in chunk:
                # need ingest timestamp now, and source/etl null
                s.add(SalesTransaction(
                    txn_date=r["txn_date"],
                    product_id=r["product_id"],
                    quantity=r["quantity"],
                    unit_price=r["unit_price"],
                    discount=r["discount"],
                    total_amount=r["total_amount"],
                    channel=r["channel"],
                    region=r["region"],
                    row_hash=r["row_hash"],
                    org_id=org_id,
                ))
            await s.flush()
        await s.commit()
        print(f"Inserted {len(sales_rows)} sales for {business['name']}")

        for chunk_start in range(0, len(expense_rows), 2000):
            chunk = expense_rows[chunk_start: chunk_start+2000]
            for r in chunk:
                s.add(Expense(
                    expense_date=r["expense_date"],
                    category=r["category"],
                    amount=r["amount"],
                    department=r.get("department"),
                    row_hash=f"{org_id.hex[:4]}-exp-{uuid.uuid4().hex[:14]}",
                    org_id=org_id,
                ))
            await s.flush()
        await s.commit()
        print(f"Inserted {len(expense_rows)} expenses for {business['name']}")

        # inventory: monthly snapshots
        from collections import defaultdict
        sales_by_month_sku = defaultdict(int)
        for r in sales_rows:
            month = r["txn_date"].replace(day=1)
            sales_by_month_sku[(month, r["product_id"])] += r["quantity"]
        stock = {sku: int(np.random.default_rng(business['seed']+1).integers(400,900)) for sku,_ in prod_map.items()}
        # map product_id -> sku for lookup
        id_to_sku = {v:k for k,v in prod_map.items()}
        months = sorted(set(m for m,_ in sales_by_month_sku.keys()))
        inv_rows = []
        for month in months:
            snap = (month + pd.offsets.MonthEnd(0)).date() if month + pd.offsets.MonthEnd(0) <= pd.Timestamp(END) else END
            for pid in prod_map.values():
                sku = id_to_sku[pid]
                sold = sales_by_month_sku.get((month, pid), 0)
                restock = int(sold * float(np.random.default_rng(business['seed']+2).uniform(0.85,1.2)))
                stock[sku] = max(0, stock[sku] - sold + restock)
                inv_rows.append({
                    "snapshot_date": snap,
                    "product_id": pid,
                    "quantity_on_hand": stock[sku],
                    "reorder_level": 120,
                    "warehouse": "main",
                })
        for r in inv_rows:
            s.add(InventoryLevel(snapshot_date=r["snapshot_date"], product_id=r["product_id"], quantity_on_hand=r["quantity_on_hand"], reorder_level=r["reorder_level"], warehouse=r["warehouse"], org_id=org_id))
        await s.commit()
        print(f"Inserted {len(inv_rows)} inventory snapshots for {business['name']}")

        # rebuild kpi snapshots per org
        min_d = min(r["txn_date"] for r in sales_rows) if sales_rows else START
        max_d = max(r["txn_date"] for r in sales_rows) if sales_rows else END
        await rebuild_kpi_snapshots(s, min_d, max_d, org_id=org_id)
        await s.commit()
        print(f"Rebuilt kpi_snapshots for {business['name']} {min_d} -> {max_d}")

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Regenerate even if business already has data")
    args = parser.parse_args()

    print("Seeding 2 businesses (isolated) ...")
    orgs = []
    for biz in BUSINESSES:
        org = await ensure_org(biz["name"], biz["slug"])
        orgs.append((biz, org))

    # users
    for biz, org in orgs:
        for role, pw, title in ROLES:
            email = f"{role}@{biz['email_domain']}"
            full_name = f"{title} ({biz['name'].split()[0]})"
            await ensure_user(email, pw, role, full_name, org.id, department=title.split()[-1] if title else "General")

    # data
    for biz, org in orgs:
        await seed_business_data(biz, org.id, force=args.force)

    print("\n" + "="*70)
    print("Seed complete. Credentials (isolated per business):")
    print("="*70)
    for biz, org in orgs:
        print(f"\nBusiness: {biz['name']}  slug={biz['slug']}  id={org.id}")
        for role, pw, _ in ROLES:
            email = f"{role}@{biz['email_domain']}"
            print(f"  {role:9} {email:35}  password: {pw}")
        print(f"  → Login via UI: /login  or  Register Business is NOT needed (already seeded)")
        print(f"  → Analytics: /dashboard (KPIs are org-scoped immediately)")
        print(f"  → Upload more data: /dashboard/data (rows will be tagged org_id={str(org.id)[:8]}...)")
    print("\nPlatform super-admin (if ADMIN_EMAIL set) sees all orgs via OrgSwitcher.")
    print("Each business's analytics/AI are strictly isolated — verify by logging in as analyst@himalayan.example.com vs analyst@everest.example.com.")
    print("="*70)
    # also print summary counts
    async with get_session_factory()() as s:
        for biz, org in orgs:
            sc = (await s.execute(select(func.count()).select_from(SalesTransaction).where(SalesTransaction.org_id==org.id))).scalar_one()
            ec = (await s.execute(select(func.count()).select_from(Expense).where(Expense.org_id==org.id))).scalar_one()
            kc = (await s.execute(select(func.count()).select_from(text("kpi_snapshots").table_valued() if False else select(func.count()).select_from(text("kpi_snapshots")).where(text("org_id = :oid")).params(oid=str(org.id))))).scalar_one() if False else 0
            # kpi count via raw
            kc2 = (await s.execute(text("SELECT count(*) FROM kpi_snapshots WHERE org_id = :oid"), {"oid": str(org.id)})).scalar_one()
            print(f"{biz['slug']}: sales={sc} expenses={ec} kpi_snapshots={kc2}")

if __name__ == "__main__":
    asyncio.run(main())
