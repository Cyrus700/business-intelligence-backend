"""add ingested_at to fact tables

Separates the *business* date of a row (txn_date / expense_date /
snapshot_date) from the moment it was loaded. Without this the system cannot
answer "what did we upload today?" — a backdated CSV imported this morning is
indistinguishable from data that has been in the warehouse for months.

Existing rows are backfilled from their ETL job's start time where the link
exists, so historical uploads keep a truthful ingestion date; anything with no
job (seeded/demo rows, inventory, which carries no etl_job_id) falls back to
the row's business date so it never claims to have been uploaded "now".

Revision ID: f2a7c9d41b30
Revises: f3a1c9b7d2e4
Create Date: 2026-08-09
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "f2a7c9d41b30"
down_revision: str | None = "f3a1c9b7d2e4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# table -> (business date column, has etl_job_id link)
TABLES: dict[str, tuple[str, bool]] = {
    "sales_transactions": ("txn_date", True),
    "expenses": ("expense_date", True),
    "inventory_levels": ("snapshot_date", False),
}


def upgrade() -> None:
    for table, (date_col, has_job) in TABLES.items():
        op.add_column(
            table,
            sa.Column(
                "ingested_at",
                sa.DateTime(),
                nullable=True,  # tightened to NOT NULL after the backfill below
                server_default=sa.func.now(),
            ),
        )

        if has_job:
            # Prefer the ETL job's start time — that is literally when the
            # upload ran.
            op.execute(
                f"""
                UPDATE {table} AS t
                   SET ingested_at = j.started_at
                  FROM etl_jobs AS j
                 WHERE t.etl_job_id = j.id
                   AND j.started_at IS NOT NULL
                """
            )

        # Everything still unset: fall back to the business date at midnight,
        # which is always <= the true ingestion time and never in the future.
        op.execute(f"UPDATE {table} SET ingested_at = {date_col}::timestamp WHERE ingested_at IS NULL")

        op.alter_column(table, "ingested_at", nullable=False)
        op.create_index(f"ix_{_index_stem(table)}_ingested_at", table, ["ingested_at"])


def downgrade() -> None:
    for table in TABLES:
        op.drop_index(f"ix_{_index_stem(table)}_ingested_at", table_name=table)
        op.drop_column(table, "ingested_at")


def _index_stem(table: str) -> str:
    """Index names in this schema use short table stems (see 2e93fabef7dd)."""
    return {
        "sales_transactions": "sales",
        "expenses": "expenses",
        "inventory_levels": "inventory",
    }[table]
