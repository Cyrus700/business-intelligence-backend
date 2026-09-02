"""Platform snapshot for the public landing page.

``GET /landing/live`` is unauthenticated, so this returns **only platform-scale
plumbing figures**: how many rows the pipelines landed, how many sources are
connected, how healthy the ETL runs are, how many insights the AI has written.

No business numbers — revenue, orders, margins, forecasts, anomalies and
customer/product dimensions live behind auth in the dashboard. If a field would
tell a stranger how much money the business makes, it does not belong here.

The payload is deliberately tiny and served from a short-lived process cache so
the landing page renders its real numbers immediately instead of flipping from
placeholder values once a slow query returns.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.clock import business_now
from app.models import (
    DataSource,
    EtlJob,
    Expense,
    Insight,
    InventoryLevel,
    SalesTransaction,
)

# The landing figures move at ETL cadence (minutes at best), so a minute of
# staleness is invisible to visitors and turns a burst of page loads into at
# most one pair of queries.
CACHE_TTL_SECONDS = 60

_cache: tuple[float, dict[str, Any]] | None = None
_lock = asyncio.Lock()


def _count(column) -> Any:
    """``(SELECT count(col))`` as a scalar subquery, for one-round-trip counts."""
    return select(func.count(column)).scalar_subquery()


async def _query_snapshot(db: AsyncSession) -> dict[str, Any]:
    # Two round trips total: one for the counters, one for pipeline health.
    counts = (
        await db.execute(
            select(
                _count(SalesTransaction.id).label("sales_rows"),
                _count(Expense.id).label("expense_rows"),
                _count(InventoryLevel.id).label("inventory_rows"),
                _count(DataSource.id).label("data_sources"),
                _count(EtlJob.id).label("etl_jobs"),
                _count(Insight.id).label("insights"),
            )
        )
    ).one()

    status_rows = (
        await db.execute(
            select(
                EtlJob.status,
                func.count(EtlJob.id).label("runs"),
                func.max(EtlJob.finished_at).label("last_finished"),
            ).group_by(EtlJob.status)
        )
    ).all()

    by_status = {str(r.status): int(r.runs) for r in status_rows}
    finished = [r.last_finished for r in status_rows if r.last_finished is not None]
    total_runs = sum(by_status.values())
    succeeded = by_status.get("succeeded", 0) + by_status.get("success", 0)
    last_run = max(finished) if finished else None

    return {
        "generated_at": business_now().isoformat(),
        "totals": {
            # "records unified" is the headline ETL number: every fact row the
            # pipelines have landed in the warehouse.
            "records_unified": int(counts.sales_rows or 0)
            + int(counts.expense_rows or 0)
            + int(counts.inventory_rows or 0),
            "data_sources": int(counts.data_sources or 0),
            "etl_jobs": int(counts.etl_jobs or 0),
            "insights": int(counts.insights or 0),
        },
        "pipeline": {
            "by_status": by_status,
            "success_rate_pct": round(succeeded / total_runs * 100, 1) if total_runs else 0.0,
            "last_run_at": last_run.isoformat() if last_run else None,
        },
    }


async def build_live_metrics(db: AsyncSession) -> dict[str, Any]:
    """Return the public platform snapshot, cached for ``CACHE_TTL_SECONDS``.

    Concurrent callers share one query pass: the first waiter runs it, the rest
    read the cache it just filled. If the warehouse errors while a previous
    snapshot is still held, that snapshot is served rather than failing the
    landing page.
    """
    global _cache

    cached = _cache
    now = time.monotonic()
    if cached and now - cached[0] < CACHE_TTL_SECONDS:
        return cached[1]

    async with _lock:
        cached = _cache
        now = time.monotonic()
        if cached and now - cached[0] < CACHE_TTL_SECONDS:
            return cached[1]
        try:
            snapshot = await _query_snapshot(db)
        except Exception:
            if cached:
                return cached[1]
            raise
        _cache = (time.monotonic(), snapshot)
        return snapshot
