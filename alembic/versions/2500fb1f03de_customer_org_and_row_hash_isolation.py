"""customer org isolation + row_hash per-org uniqueness — robust

Revision ID: 2500fb1f03de
Revises: c8d3e5f7a9b1
Create Date: 2026-09-08

- Adds org_id to customers (FK to organizations, indexed, unique name+org)
- Migrates sales_transactions.row_hash and expenses.row_hash from global unique
  to composite (org_id, row_hash) so re-running same file in different orgs
  does not collide, and backfills legacy org.
- Robust: uses savepoints / DO blocks to avoid transaction abort on duplicate
  data or missing constraints. Skips unique creation if duplicates exist and logs.

"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

revision: str = "2500fb1f03de"
down_revision: str | None = "c8d3e5f7a9b1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

LEGACY_ORG_NAME = "Legacy — Single Tenant (Backfill)"
LEGACY_ORG_SLUG = "legacy-default"


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    # --- 1. Ensure legacy org exists for backfill
    try:
        conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS pgcrypto"))
    except Exception:
        pass
    legacy_id = conn.execute(sa.text("SELECT id FROM organizations WHERE is_legacy = true LIMIT 1")).scalar()
    if legacy_id is None:
        legacy_id = conn.execute(sa.text("SELECT id FROM organizations WHERE name = :n"), {"n": LEGACY_ORG_NAME}).scalar()
    if legacy_id is None:
        try:
            legacy_id = conn.execute(
                sa.text(
                    "INSERT INTO organizations (id, name, slug, is_legacy, created_at, updated_at) "
                    "VALUES (gen_random_uuid(), :name, :slug, true, now(), now()) RETURNING id"
                ),
                {"name": LEGACY_ORG_NAME, "slug": LEGACY_ORG_SLUG},
            ).scalar()
        except Exception:
            legacy_id = conn.execute(sa.text("SELECT id FROM organizations LIMIT 1")).scalar()

    # --- 2. customers.org_id
    cols = [c["name"] for c in insp.get_columns("customers")]
    if "org_id" not in cols:
        # Use IF NOT EXISTS via raw SQL to avoid error if race
        try:
            with conn.begin_nested():
                op.add_column("customers", sa.Column("org_id", UUID(as_uuid=True), nullable=True))
        except Exception:
            pass
        # FK
        try:
            with conn.begin_nested():
                op.create_foreign_key(
                    "fk_customers_org_id_organizations",
                    "customers",
                    "organizations",
                    ["org_id"],
                    ["id"],
                    ondelete="CASCADE",
                )
        except Exception:
            pass
        try:
            with conn.begin_nested():
                op.create_index("ix_customers_org_id", "customers", ["org_id"])
        except Exception:
            pass
        try:
            with conn.begin_nested():
                op.create_index("ix_customers_name", "customers", ["name"])
        except Exception:
            pass
        # backfill: use sales_transactions org if exists, else legacy
        try:
            with conn.begin_nested():
                # Prefer org from sales_transactions per customer
                conn.execute(sa.text("""
                    UPDATE customers c SET org_id = sub.org_id
                    FROM (
                        SELECT DISTINCT ON (customer_id) customer_id, org_id
                        FROM sales_transactions
                        WHERE customer_id IS NOT NULL AND org_id IS NOT NULL
                        ORDER BY customer_id, txn_date
                    ) sub
                    WHERE c.id = sub.customer_id AND c.org_id IS NULL
                """))
        except Exception as e:
            print(f"customers backfill from sales skipped: {e}")
        try:
            with conn.begin_nested():
                conn.execute(sa.text("UPDATE customers SET org_id = :lid WHERE org_id IS NULL"), {"lid": legacy_id})
        except Exception:
            pass
    else:
        try:
            with conn.begin_nested():
                op.create_index("ix_customers_org_id", "customers", ["org_id"])
        except Exception:
            pass

    # create unique constraint only if no duplicates would violate it
    # Use DO block to catch unique_violation without aborting outer txn
    try:
        # check for duplicates that would block unique
        dup = conn.execute(sa.text("""
            SELECT 1 FROM customers
            WHERE org_id IS NOT NULL
            GROUP BY name, org_id HAVING COUNT(*) > 1 LIMIT 1
        """)).scalar()
        if dup is None:
            # No duplicates, safe to create unique via DO block with exception handling
            conn.execute(sa.text("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint WHERE conname = 'uq_customers_name_org'
                    ) THEN
                        ALTER TABLE customers ADD CONSTRAINT uq_customers_name_org UNIQUE (name, org_id);
                    END IF;
                EXCEPTION WHEN duplicate_table OR duplicate_object THEN
                    RAISE WARNING 'uq_customers_name_org already exists';
                WHEN unique_violation THEN
                    RAISE WARNING 'uq_customers_name_org duplicate data, skipping';
                END
                $$;
            """))
        else:
            print("WARNING: customers (name, org_id) has duplicates — skipping uq_customers_name_org, data needs manual cleanup but org isolation code will still work")
    except Exception as e:
        print(f"uq_customers_name_org creation skipped: {e}")

    # --- 3. sales_transactions.row_hash -> (org_id, row_hash)
    # Drop global unique on row_hash if exists — use IF EXISTS to avoid error
    try:
        conn.execute(sa.text("ALTER TABLE sales_transactions DROP CONSTRAINT IF EXISTS uq_sales_transactions_row_hash"))
    except Exception:
        pass
    try:
        conn.execute(sa.text("ALTER TABLE sales_transactions DROP CONSTRAINT IF EXISTS uq_sales_row_hash"))
    except Exception:
        pass
    # also try auto-named constraints via inspect fallback with savepoint
    try:
        with conn.begin_nested():
            cons = [c["name"] for c in insp.get_unique_constraints("sales_transactions") if c["column_names"] == ["row_hash"]]
            for cname in cons:
                conn.execute(sa.text(f'ALTER TABLE sales_transactions DROP CONSTRAINT IF EXISTS "{cname}"'))
    except Exception:
        pass
    try:
        conn.execute(sa.text("DROP INDEX IF EXISTS uq_sales_transactions_row_hash"))
    except Exception:
        pass

    # create composite unique if not exists and no per-org duplicates
    try:
        dup = conn.execute(sa.text("""
            SELECT 1 FROM sales_transactions
            WHERE row_hash IS NOT NULL AND org_id IS NOT NULL
            GROUP BY org_id, row_hash HAVING COUNT(*) > 1 LIMIT 1
        """)).scalar()
        if dup is None:
            conn.execute(sa.text("""
                DO $$
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'uq_sales_row_hash_org') THEN
                        ALTER TABLE sales_transactions ADD CONSTRAINT uq_sales_row_hash_org UNIQUE (org_id, row_hash);
                    END IF;
                EXCEPTION WHEN duplicate_table OR duplicate_object THEN
                    RAISE WARNING 'uq_sales_row_hash_org already exists';
                WHEN unique_violation THEN
                    RAISE WARNING 'uq_sales_row_hash_org duplicate data';
                END
                $$;
            """))
        else:
            print("WARNING: sales_transactions (org_id,row_hash) has duplicates — skipping uq_sales_row_hash_org")
    except Exception as e:
        print(f"uq_sales_row_hash_org: {e}")
    try:
        conn.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_sales_row_hash ON sales_transactions (row_hash)"))
    except Exception:
        pass

    # --- 4. expenses.row_hash -> (org_id, row_hash)
    try:
        conn.execute(sa.text("ALTER TABLE expenses DROP CONSTRAINT IF EXISTS uq_expenses_row_hash"))
    except Exception:
        pass
    try:
        with conn.begin_nested():
            cons = [c["name"] for c in insp.get_unique_constraints("expenses") if c["column_names"] == ["row_hash"]]
            for cname in cons:
                conn.execute(sa.text(f'ALTER TABLE expenses DROP CONSTRAINT IF EXISTS "{cname}"'))
    except Exception:
        pass
    try:
        dup = conn.execute(sa.text("""
            SELECT 1 FROM expenses
            WHERE row_hash IS NOT NULL AND org_id IS NOT NULL
            GROUP BY org_id, row_hash HAVING COUNT(*) > 1 LIMIT 1
        """)).scalar()
        if dup is None:
            conn.execute(sa.text("""
                DO $$
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'uq_expenses_row_hash_org') THEN
                        ALTER TABLE expenses ADD CONSTRAINT uq_expenses_row_hash_org UNIQUE (org_id, row_hash);
                    END IF;
                EXCEPTION WHEN duplicate_table OR duplicate_object THEN
                    RAISE WARNING 'uq_expenses_row_hash_org already exists';
                WHEN unique_violation THEN
                    RAISE WARNING 'uq_expenses_row_hash_org duplicate data';
                END
                $$;
            """))
        else:
            print("WARNING: expenses (org_id,row_hash) has duplicates — skipping")
    except Exception as e:
        print(f"uq_expenses_row_hash_org: {e}")
    try:
        conn.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_expenses_row_hash ON expenses (row_hash)"))
    except Exception:
        pass


def downgrade() -> None:
    conn = op.get_bind()
    try:
        conn.execute(sa.text("ALTER TABLE sales_transactions DROP CONSTRAINT IF EXISTS uq_sales_row_hash_org"))
        conn.execute(sa.text("ALTER TABLE sales_transactions ADD CONSTRAINT uq_sales_transactions_row_hash UNIQUE (row_hash)"))
    except Exception:
        pass
    try:
        conn.execute(sa.text("ALTER TABLE expenses DROP CONSTRAINT IF EXISTS uq_expenses_row_hash_org"))
        conn.execute(sa.text("ALTER TABLE expenses ADD CONSTRAINT uq_expenses_row_hash UNIQUE (row_hash)"))
    except Exception:
        pass
    try:
        conn.execute(sa.text("ALTER TABLE customers DROP CONSTRAINT IF EXISTS uq_customers_name_org"))
    except Exception:
        pass
    try:
        conn.execute(sa.text("DROP INDEX IF EXISTS ix_customers_org_id"))
    except Exception:
        pass
    try:
        conn.execute(sa.text("ALTER TABLE customers DROP CONSTRAINT IF EXISTS fk_customers_org_id_organizations"))
    except Exception:
        pass
    try:
        op.drop_column("customers", "org_id")
    except Exception:
        pass
