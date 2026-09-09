"""watermark per-org — make data_watermarks org-scoped

Revision ID: 7303a1912965
Revises: 2500fb1f03de
Create Date: 2026-09-08

- Adds org_id to data_watermarks (FK to organizations, unique)
- Drops single-row check constraint (id = 1) to allow per-org rows
- Backfills existing global row to legacy org if present
- Creates index/unique on org_id

"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

revision: str = "7303a1912965"
down_revision: str | None = "2500fb1f03de"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    # ensure table exists (migration 6ad649...)
    tables = insp.get_table_names()
    if "data_watermarks" not in tables:
        return

    cols = [c["name"] for c in insp.get_columns("data_watermarks")]
    if "org_id" not in cols:
        op.add_column("data_watermarks", sa.Column("org_id", UUID(as_uuid=True), nullable=True))
        try:
            op.create_foreign_key(
                "fk_data_watermarks_org_id_organizations",
                "data_watermarks",
                "organizations",
                ["org_id"],
                ["id"],
                ondelete="CASCADE",
            )
        except Exception:
            pass
        try:
            op.create_index("ix_data_watermarks_org_id", "data_watermarks", ["org_id"], unique=True)
        except Exception:
            pass
        try:
            op.create_unique_constraint("uq_data_watermarks_org_id", "data_watermarks", ["org_id"])
        except Exception:
            pass

        # Drop single-row check if exists
        try:
            conn.execute(sa.text("ALTER TABLE data_watermarks DROP CONSTRAINT IF EXISTS single_watermark_row"))
        except Exception:
            pass

        # Migrate existing global watermark (id=1, org_id NULL) to legacy org or keep as global fallback
        try:
            legacy_id = conn.execute(sa.text("SELECT id FROM organizations WHERE is_legacy = true LIMIT 1")).scalar()
            if legacy_id is not None:
                # If there's a global row with NULL org, duplicate it for legacy org, keep original as fallback?
                # For simplicity, assign NULL org row to legacy org where org_id IS NULL and id=1
                conn.execute(sa.text("UPDATE data_watermarks SET org_id = :lid WHERE org_id IS NULL AND id = 1"), {"lid": legacy_id})
                # If multiple rows with NULL, assign them too
                conn.execute(sa.text("UPDATE data_watermarks SET org_id = :lid WHERE org_id IS NULL"), {"lid": legacy_id})
        except Exception as e:
            print(f"watermark backfill skipped: {e}")

        # Make org_id nullable for super_admin global view? Keep nullable to allow global fallback.
        # But for strict per-org, we keep nullable and allow multiple rows with distinct orgs.

    # Ensure constraint for id primary still exists; we keep id as PK but allow multiple rows with different ids per org
    # Change primary maybe not needed; we just need org_id unique.
    # Ensure we have at least index for org_id filter
    try:
        op.create_index("ix_data_watermarks_org_id", "data_watermarks", ["org_id"])
    except Exception:
        pass


def downgrade() -> None:
    try:
        op.drop_constraint("uq_data_watermarks_org_id", "data_watermarks", type_="unique")
    except Exception:
        pass
    try:
        op.drop_index("ix_data_watermarks_org_id", table_name="data_watermarks")
    except Exception:
        pass
    try:
        op.drop_constraint("fk_data_watermarks_org_id_organizations", "data_watermarks", type_="foreignkey")
    except Exception:
        pass
    try:
        op.drop_column("data_watermarks", "org_id")
    except Exception:
        pass
    try:
        op.create_check_constraint("single_watermark_row", "data_watermarks", "id = 1")
    except Exception:
        pass
