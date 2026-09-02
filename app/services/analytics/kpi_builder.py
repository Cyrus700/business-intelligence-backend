"""Rebuild kpi_snapshots for a date range — the dashboards' fast read path.

Called after every successful load for the affected dates, so headline KPIs
never scan the fact tables (docs/03-database-schema.md indexing notes).
"""

from datetime import date

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

_REBUILD_SQL = """
DELETE FROM kpi_snapshots WHERE snapshot_date BETWEEN :d1 AND :d2 AND org_id = :org_id;

INSERT INTO kpi_snapshots (snapshot_date, metric, dimensions, value, org_id)
SELECT txn_date, 'revenue', '{}'::jsonb, SUM(total_amount), :org_id
FROM sales_transactions WHERE txn_date BETWEEN :d1 AND :d2 AND org_id = :org_id GROUP BY txn_date;

INSERT INTO kpi_snapshots (snapshot_date, metric, dimensions, value, org_id)
SELECT txn_date, 'revenue', jsonb_build_object('region', region), SUM(total_amount), :org_id
FROM sales_transactions
WHERE txn_date BETWEEN :d1 AND :d2 AND region IS NOT NULL AND org_id = :org_id
GROUP BY txn_date, region;

INSERT INTO kpi_snapshots (snapshot_date, metric, dimensions, value, org_id)
SELECT txn_date, 'revenue', jsonb_build_object('channel', channel), SUM(total_amount), :org_id
FROM sales_transactions
WHERE txn_date BETWEEN :d1 AND :d2 AND channel IS NOT NULL AND org_id = :org_id
GROUP BY txn_date, channel;

INSERT INTO kpi_snapshots (snapshot_date, metric, dimensions, value, org_id)
SELECT txn_date, 'orders', '{}'::jsonb, COUNT(*), :org_id
FROM sales_transactions WHERE txn_date BETWEEN :d1 AND :d2 AND org_id = :org_id GROUP BY txn_date;

INSERT INTO kpi_snapshots (snapshot_date, metric, dimensions, value, org_id)
SELECT txn_date, 'avg_order_value', '{}'::jsonb, AVG(total_amount), :org_id
FROM sales_transactions WHERE txn_date BETWEEN :d1 AND :d2 AND org_id = :org_id GROUP BY txn_date;

INSERT INTO kpi_snapshots (snapshot_date, metric, dimensions, value, org_id)
SELECT txn_date, 'gross_margin', '{}'::jsonb,
       SUM(s.total_amount - COALESCE(p.unit_cost, 0) * s.quantity), :org_id
FROM sales_transactions s LEFT JOIN products p ON p.id = s.product_id
WHERE txn_date BETWEEN :d1 AND :d2 AND s.org_id = :org_id GROUP BY txn_date;

INSERT INTO kpi_snapshots (snapshot_date, metric, dimensions, value, org_id)
SELECT expense_date, 'expense_total', '{}'::jsonb, SUM(amount), :org_id
FROM expenses WHERE expense_date BETWEEN :d1 AND :d2 AND org_id = :org_id GROUP BY expense_date;

INSERT INTO kpi_snapshots (snapshot_date, metric, dimensions, value, org_id)
SELECT expense_date, 'expense_total', jsonb_build_object('category', category), SUM(amount), :org_id
FROM expenses WHERE expense_date BETWEEN :d1 AND :d2 AND org_id = :org_id GROUP BY expense_date, category;

INSERT INTO kpi_snapshots (snapshot_date, metric, dimensions, value, org_id)
SELECT snapshot_date, 'stockout_count', '{}'::jsonb,
       COUNT(*) FILTER (WHERE quantity_on_hand <= reorder_level), :org_id
FROM inventory_levels WHERE snapshot_date BETWEEN :d1 AND :d2 AND org_id = :org_id GROUP BY snapshot_date;
"""


async def rebuild_kpi_snapshots(db: AsyncSession, start: date, end: date, org_id=None) -> None:
    if org_id is None:
        # No org context (e.g. super_admin global view or legacy call) — skip per-org rebuild
        # Fallback: rebuild without org filter for backwards compat if no org_id provided
        # But per-tenant mode requires org_id; log warning and run legacy SQL without org
        for statement in _REBUILD_SQL.split(";"):
            if statement.strip():
                # For legacy path, strip org_id condition (not ideal, but keeps old behavior)
                legacy_stmt = (
                    statement.replace(" AND org_id = :org_id", "").replace(", org_id", "").replace(", :org_id", "")
                )
                # Need to handle DELETE without org
                if "DELETE" in legacy_stmt:
                    legacy_stmt = legacy_stmt.replace(" AND org_id = :org_id", "")
                await db.execute(text(legacy_stmt), {"d1": start, "d2": end})
        return
    for statement in _REBUILD_SQL.split(";"):
        if statement.strip():
            await db.execute(text(statement), {"d1": start, "d2": end, "org_id": str(org_id)})
