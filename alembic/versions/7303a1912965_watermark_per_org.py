"""watermark per-org — make data_watermarks org-scoped

Revision ID: 7303a1912965
Revises: 2500fb1f03de
Create Date: 2026-09-08

- Adds org_id to data_watermarks (FK to organizations, unique)
- Drops single-row check constraint (id = 1) to allow per-org rows
- Backfills existing global row to legacy org if present
- Creates index/unique on org_id — robust via DO blocks

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

    tables = insp.get_table_names()
    if "data_watermarks" not in tables:
        return

    cols = [c["name"] for c in insp.get_columns("data_watermarks")]
    if "org_id" not in cols:
        try:
            with conn.begin_nested():
                op.add_column("data_watermarks", sa.Column("org_id", UUID(as_uuid=True), nullable=True))
        except Exception:
            pass
        try:
            with conn.begin_nested():
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
        # Drop single-row check if exists — use IF EXISTS
        try:
            conn.execute(sa.text("ALTER TABLE data_watermarks DROP CONSTRAINT IF EXISTS single_watermark_row"))
        except Exception:
            pass
        # Backfill existing global watermark to legacy org before creating unique
        try:
            legacy_id = conn.execute(sa.text("SELECT id FROM organizations WHERE is_legacy = true LIMIT 1")).scalar()
            if legacy_id is not None:
                conn.execute(sa.text("UPDATE data_watermarks SET org_id = :lid WHERE org_id IS NULL AND id = 1"), {"lid": legacy_id})
                conn.execute(sa.text("UPDATE data_watermarks SET org_id = :lid WHERE org_id IS NULL"), {"lid": legacy_id})
        except Exception as e:
            print(f"watermark backfill skipped: {e}")

        # Create unique/index — allow multiple NULLs in PG, so safe; use DO block
        try:
            dup = conn.execute(sa.text("SELECT 1 FROM data_watermarks WHERE org_id IS NOT NULL GROUP BY org_id HAVING COUNT(*) >1 LIMIT 1")).scalar()
            if dup is None:
                conn.execute(sa.text("""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'uq_data_watermarks_org_id') THEN
                            ALTER TABLE data_watermarks ADD CONSTRAINT uq_data_watermarks_org_id UNIQUE (org_id);
                        END IF;
                    EXCEPTION WHEN duplicate_table OR duplicate_object THEN
                        RAISE WARNING 'uq_data_watermarks_org_id exists';
                    WHEN unique_violation THEN
                        RAISE WARNING 'watermark org_id duplicate';
                    END
                    $$;
                """))
            else:
                print("WARNING: data_watermarks duplicate org_id — skipping unique")
        except Exception as e:
            print(f"uq_data_watermarks_org_id: {e}")

        try:
            conn.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_data_watermarks_org_id ON data_watermarks (org_id)"))
        except Exception:
            pass
    else:
        # Ensure index exists
        try:
            conn.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_data_watermarks_org_id ON data_watermarks (org_id)"))
        except Exception:
            pass
        # Ensure single_watermark_row dropped
        try:
            conn.execute(sa.text("ALTER TABLE data_watermarks DROP CONSTRAINT IF EXISTS single_watermark_row"))
        except Exception:
            pass


def downgrade() -> None:
    conn = op.get_bind()
    try:
        conn.execute(sa.text("ALTER TABLE data_watermarks DROP CONSTRAINT IF EXISTS uq_data_watermarks_org_id"))
    except Exception:
        pass
    try:
        conn.execute(sa.text("DROP INDEX IF EXISTS ix_data_watermarks_org_id"))
    except Exception:
        pass
    try:
        conn.execute(sa.text("ALTER TABLE data_watermarks DROP CONSTRAINT IF EXISTS fk_data_watermarks_org_id_organizations"))
    except Exception:
        pass
    try:
        op.drop_column("data_watermarks", "org_id")
    except Exception:
        pass
    try:
        conn.execute(sa.text("ALTER TABLE data_watermarks ADD CONSTRAINT single_watermark_row CHECK (id = 1)"))
    except Exception:
        pass
