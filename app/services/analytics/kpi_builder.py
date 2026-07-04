"""Rebuild kpi_snapshots for a date range — the dashboards' fast read path.

Called after every successful load for the affected dates, so headline KPIs
never scan the fact tables (docs/03-database-schema.md indexing notes).
"""

from datetime import date

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

_REBUILD_SQL = """
DELETE FROM kpi_snapshots WHERE snapshot_date BETWEEN :d1 AND :d2;

INSERT INTO kpi_snapshots (snapshot_date, metric, dimensions, value)
SELECT txn_date, 'revenue', '{}'::jsonb, SUM(total_amount)
FROM sales_transactions WHERE txn_date BETWEEN :d1 AND :d2 GROUP BY txn_date;

INSERT INTO kpi_snapshots (snapshot_date, metric, dimensions, value)
SELECT txn_date, 'revenue', jsonb_build_object('region', region), SUM(total_amount)
FROM sales_transactions
WHERE txn_date BETWEEN :d1 AND :d2 AND region IS NOT NULL
GROUP BY txn_date, region;

INSERT INTO kpi_snapshots (snapshot_date, metric, dimensions, value)
SELECT txn_date, 'revenue', jsonb_build_object('channel', channel), SUM(total_amount)
FROM sales_transactions
WHERE txn_date BETWEEN :d1 AND :d2 AND channel IS NOT NULL
GROUP BY txn_date, channel;

INSERT INTO kpi_snapshots (snapshot_date, metric, dimensions, value)
SELECT txn_date, 'orders', '{}'::jsonb, COUNT(*)
FROM sales_transactions WHERE txn_date BETWEEN :d1 AND :d2 GROUP BY txn_date;

INSERT INTO kpi_snapshots (snapshot_date, metric, dimensions, value)
SELECT txn_date, 'avg_order_value', '{}'::jsonb, AVG(total_amount)
FROM sales_transactions WHERE txn_date BETWEEN :d1 AND :d2 GROUP BY txn_date;

INSERT INTO kpi_snapshots (snapshot_date, metric, dimensions, value)
SELECT txn_date, 'gross_margin', '{}'::jsonb,
       SUM(s.total_amount - COALESCE(p.unit_cost, 0) * s.quantity)
FROM sales_transactions s LEFT JOIN products p ON p.id = s.product_id
WHERE txn_date BETWEEN :d1 AND :d2 GROUP BY txn_date;

INSERT INTO kpi_snapshots (snapshot_date, metric, dimensions, value)
SELECT expense_date, 'expense_total', '{}'::jsonb, SUM(amount)
FROM expenses WHERE expense_date BETWEEN :d1 AND :d2 GROUP BY expense_date;

INSERT INTO kpi_snapshots (snapshot_date, metric, dimensions, value)
SELECT expense_date, 'expense_total', jsonb_build_object('category', category), SUM(amount)
FROM expenses WHERE expense_date BETWEEN :d1 AND :d2 GROUP BY expense_date, category;

INSERT INTO kpi_snapshots (snapshot_date, metric, dimensions, value)
SELECT snapshot_date, 'stockout_count', '{}'::jsonb,
       COUNT(*) FILTER (WHERE quantity_on_hand <= reorder_level)
FROM inventory_levels WHERE snapshot_date BETWEEN :d1 AND :d2 GROUP BY snapshot_date;
"""


async def rebuild_kpi_snapshots(db: AsyncSession, start: date, end: date) -> None:
    for statement in _REBUILD_SQL.split(";"):
        if statement.strip():
            await db.execute(text(statement), {"d1": start, "d2": end})
