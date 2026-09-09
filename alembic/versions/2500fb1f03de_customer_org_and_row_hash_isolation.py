"""customer org isolation + row_hash per-org uniqueness

Revision ID: 2500fb1f03de
Revises: c8d3e5f7a9b1
Create Date: 2026-09-08

- Adds org_id to customers (FK to organizations, indexed, unique name+org)
- Migrates sales_transactions.row_hash and expenses.row_hash from global unique
  to composite (org_id, row_hash) so re-running same file in different orgs
  does not collide, and backfills legacy org.
- Keeps backwards compatible: if column/constraint already exists, skip.

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
        op.add_column("customers", sa.Column("org_id", UUID(as_uuid=True), nullable=True))
        try:
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
            op.create_index("ix_customers_org_id", "customers", ["org_id"])
        except Exception:
            pass
        try:
            op.create_index("ix_customers_name", "customers", ["name"])
        except Exception:
            pass
        # backfill to legacy org
        try:
            conn.execute(sa.text("UPDATE customers SET org_id = :lid WHERE org_id IS NULL"), {"lid": legacy_id})
        except Exception:
            pass
        # unique per org — allow NULL org for pre-migration if any, but normally all filled
        try:
            op.create_unique_constraint("uq_customers_name_org", "customers", ["name", "org_id"])
        except Exception as e:
            if "already exists" not in str(e):
                print(f"uq_customers_name_org creation skipped: {e}")
    else:
        # ensure indexes exist
        try:
            op.create_index("ix_customers_org_id", "customers", ["org_id"])
        except Exception:
            pass
        try:
            op.create_unique_constraint("uq_customers_name_org", "customers", ["name", "org_id"])
        except Exception:
            pass

    # --- 3. sales_transactions.row_hash -> (org_id, row_hash)
    # Drop global unique on row_hash if exists
    try:
        # name from initial schema is uq_sales_transactions_row_hash
        op.drop_constraint("uq_sales_transactions_row_hash", "sales_transactions", type_="unique")
    except Exception:
        pass
    # also try auto-named constraints
    try:
        cons = [c["name"] for c in insp.get_unique_constraints("sales_transactions") if c["column_names"] == ["row_hash"]]
        for cname in cons:
            op.drop_constraint(cname, "sales_transactions", type_="unique")
    except Exception:
        pass
    try:
        op.drop_index("uq_sales_transactions_row_hash", table_name="sales_transactions")
    except Exception:
        pass
    # create composite unique if not exists
    try:
        existing_uq = [c["name"] for c in insp.get_unique_constraints("sales_transactions")]
        if "uq_sales_row_hash_org" not in existing_uq:
            op.create_unique_constraint("uq_sales_row_hash_org", "sales_transactions", ["org_id", "row_hash"])
    except Exception as e:
        if "already exists" not in str(e):
            print(f"uq_sales_row_hash_org: {e}")
    try:
        op.create_index("ix_sales_row_hash", "sales_transactions", ["row_hash"])
    except Exception:
        pass

    # --- 4. expenses.row_hash -> (org_id, row_hash)
    try:
        op.drop_constraint("uq_expenses_row_hash", "expenses", type_="unique")
    except Exception:
        pass
    try:
        cons = [c["name"] for c in insp.get_unique_constraints("expenses") if c["column_names"] == ["row_hash"]]
        for cname in cons:
            op.drop_constraint(cname, "expenses", type_="unique")
    except Exception:
        pass
    try:
        existing_uq = [c["name"] for c in insp.get_unique_constraints("expenses")]
        if "uq_expenses_row_hash_org" not in existing_uq:
            op.create_unique_constraint("uq_expenses_row_hash_org", "expenses", ["org_id", "row_hash"])
    except Exception as e:
        if "already exists" not in str(e):
            print(f"uq_expenses_row_hash_org: {e}")
    try:
        op.create_index("ix_expenses_row_hash", "expenses", ["row_hash"])
    except Exception:
        pass


def downgrade() -> None:
    try:
        op.drop_constraint("uq_sales_row_hash_org", "sales_transactions", type_="unique")
        op.create_unique_constraint("uq_sales_transactions_row_hash", "sales_transactions", ["row_hash"])
    except Exception:
        pass
    try:
        op.drop_constraint("uq_expenses_row_hash_org", "expenses", type_="unique")
        op.create_unique_constraint("uq_expenses_row_hash", "expenses", ["row_hash"])
    except Exception:
        pass
    try:
        op.drop_constraint("uq_customers_name_org", "customers", type_="unique")
    except Exception:
        pass
    try:
        op.drop_index("ix_customers_org_id", table_name="customers")
    except Exception:
        pass
    try:
        op.drop_constraint("fk_customers_org_id_organizations", "customers", type_="foreignkey")
    except Exception:
        pass
    try:
        op.drop_column("customers", "org_id")
    except Exception:
        pass
