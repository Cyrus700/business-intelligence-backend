"""Diagnostic analytics — answers *why* a metric changed.

Turns "Revenue decreased 12%" into an investigated answer:

    Revenue ↓ 12%
      Region: Kathmandu ↓ 21%  (contributed -38% of the total change)
      Product: Product B ↓ 17% (contributed -29%)
      Channel: Online  ↓ 14%  (contributed -22%)

Method — contribution (delta) decomposition: for each dimension member the
member's *absolute* delta (current − previous) is measured, then expressed as
a share of the total metric delta. The members whose deltas explain the most
of the overall movement become the primary/secondary factors in the summary.

This is pure arithmetic over the same fact tables the analytics layer uses —
no ML, no smoothing, no invented numbers. Negative contributions reduce the
total; positive ones grow it. The sum of all member deltas equals the total
delta (up to rounding), which makes the breakdown auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Expense, Product, SalesTransaction

SUPPORTED_DIMENSIONS = ("region", "channel", "category", "product")
SUPPORTED_METRICS = ("revenue", "orders", "avg_order_value", "gross_margin", "expense_total")

# How much of the total change a member must explain to be called "primary".
_PRIMARY_SHARE = 0.15


@dataclass
class MemberContribution:
    key: str
    current: float
    previous: float
    delta: float
    contribution_pct: float  # share of total metric delta explained by this member
    change_pct: float | None  # member's own % change


@dataclass
class DimensionAnalysis:
    dimension: str
    members: list[MemberContribution] = field(default_factory=list)

    @property
    def drivers(self) -> list[MemberContribution]:
        """Members pushing the metric in its actual direction, ranked."""
        sign = 1 if _overall_direction(self) != "down" else -1
        return sorted(
            (m for m in self.members if m.delta * sign > 0),
            key=lambda m: abs(m.contribution_pct),
            reverse=True,
        )

    @property
    def drags(self) -> list[MemberContribution]:
        """Members pulling against the metric's direction, ranked."""
        sign = -1 if _overall_direction(self) != "down" else 1
        return sorted(
            (m for m in self.members if m.delta * sign > 0),
            key=lambda m: abs(m.contribution_pct),
            reverse=True,
        )

    @property
    def net_contribution(self) -> float:
        return sum(m.contribution_pct for m in self.members)


def _overall_direction(dim: DimensionAnalysis) -> str:
    return "down" if sum(m.delta for m in dim.members) < 0 else "up"


async def diagnose_change(
    db: AsyncSession,
    *,
    metric: str = "revenue",
    date_from: date,
    date_to: date,
    dimensions: tuple[str, ...] = ("region", "channel", "product"),
    region: str | None = None,
    channel: str | None = None,
    category: str | None = None,
) -> dict:
    """Decompose a metric's period-over-period change by dimension."""
    metric = metric if metric in SUPPORTED_METRICS else "revenue"
    dimensions = tuple(d for d in dimensions if d in SUPPORTED_DIMENSIONS) or ("region",)

    span = (date_to - date_from).days + 1
    prev_to = date_from - timedelta(days=1)
    prev_from = prev_to - timedelta(days=span - 1)

    filters = {"region": region, "channel": channel, "category": category}

    total_current = await _metric_total(db, metric, date_from, date_to, filters)
    total_previous = await _metric_total(db, metric, prev_from, prev_to, filters)
    delta = total_current - total_previous
    change_pct = round(delta / total_previous * 100, 1) if total_previous else None

    dim_results: dict[str, dict] = {}
    for dimension in dimensions:
        members = await _member_breakdown(
            db, metric, dimension, date_from, date_to, prev_from, prev_to, filters
        )
        contributions = []
        for key, cur, prev in members:
            member_delta = cur - prev
            contribution_pct = (
                round(member_delta / delta * 100, 1) if delta else 0.0
            )
            contributions.append(
                MemberContribution(
                    key=key,
                    current=cur,
                    previous=prev,
                    delta=round(member_delta, 2),
                    contribution_pct=contribution_pct,
                    change_pct=round((cur - prev) / prev * 100, 1) if prev else None,
                )
            )
        contributions.sort(key=lambda m: abs(m.delta), reverse=True)
        analysis = DimensionAnalysis(dimension=dimension, members=contributions)
        dim_results[dimension] = {
            "members": [
                {
                    "key": m.key,
                    "current": m.current,
                    "previous": m.previous,
                    "delta": m.delta,
                    "contribution_pct": m.contribution_pct,
                    "change_pct": m.change_pct,
                }
                for m in contributions
            ],
            "drivers": [m.key for m in analysis.drivers],
            "drags": [m.key for m in analysis.drags],
            "net_contribution": round(analysis.net_contribution, 1),
        }

    summary = _build_summary(dim_results, delta, change_pct)

    return {
        "metric": metric,
        "period": {"from": date_from, "to": date_to, "span_days": span},
        "comparison": {"from": prev_from, "to": prev_to},
        "current": round(total_current, 2),
        "previous": round(total_previous, 2),
        "delta": round(delta, 2),
        "change_pct": change_pct,
        "direction": "down" if delta < 0 else "up",
        "dimensions": dim_results,
        "summary": summary,
    }


def _build_summary(
    dim_results: dict[str, dict], delta: float, change_pct: float | None
) -> dict:
    """Pick primary/secondary factors across all dimensions."""
    candidates: list[tuple[float, str, str]] = []
    for dimension, info in dim_results.items():
        for member in info["members"]:
            if member["key"] == "(unknown)":
                continue
            candidates.append((abs(member["contribution_pct"]), dimension, member["key"]))

    candidates.sort(reverse=True)
    primary = candidates[0] if candidates else None
    secondary = candidates[1] if len(candidates) > 1 else None

    direction_word = "increase" if delta >= 0 else "decline"
    primary_text = None
    secondary_text = None
    if primary and primary[0] >= _PRIMARY_SHARE * 100:
        sign = "+" if delta >= 0 else "−"
        primary_text = (
            f"{primary[2].capitalize()} ({primary[1]}) accounts for "
            f"{sign}{primary[0]:.0f}% of the {direction_word}"
        )
    if secondary and secondary[0] >= 10:
        secondary_text = (
            f"{secondary[2].capitalize()} ({secondary[1]}) accounts for "
            f"{secondary[0]:.0f}%"
        )

    return {
        "direction": "down" if delta < 0 else "up",
        "direction_word": direction_word,
        "primary_factor": primary_text,
        "secondary_factor": secondary_text,
        "primary_contributor": {
            "dimension": primary[1],
            "key": primary[2],
            "contribution_pct": round(primary[0], 1),
        }
        if primary
        else None,
        "change_pct": change_pct,
    }


async def _metric_total(
    db: AsyncSession,
    metric: str,
    date_from: date,
    date_to: date,
    filters: dict[str, str | None],
) -> float:
    if metric == "expense_total":
        value = func.sum(Expense.amount)
        stmt = (
            select(func.coalesce(value, 0))
            .where(Expense.expense_date.between(date_from, date_to))
        )
        return float((await db.execute(stmt)).scalar_one())

    value_expr, joins = _metric_expr(metric)
    conditions = [SalesTransaction.txn_date.between(date_from, date_to)]
    conditions += _sales_filters(filters)
    stmt = select(func.coalesce(value_expr, 0))
    for join in joins:
        stmt = stmt.select_from(SalesTransaction.__table__.join(*join))
    if not joins:
        stmt = stmt.select_from(SalesTransaction.__table__)
    stmt = stmt.where(and_(*conditions))
    return float((await db.execute(stmt)).scalar_one())


async def _member_breakdown(
    db: AsyncSession,
    metric: str,
    dimension: str,
    date_from: date,
    date_to: date,
    prev_from: date,
    prev_to: date,
    filters: dict[str, str | None],
) -> list[tuple[str, float, float]]:
    """Per-member (current, previous) values for a dimension."""
    if metric == "expense_total":
        key = Expense.category
        current_q = (
            select(key.label("key"), func.sum(Expense.amount).label("v"))
            .where(Expense.expense_date.between(date_from, date_to))
            .group_by(key)
        )
        previous_q = (
            select(key.label("key"), func.sum(Expense.amount).label("v"))
            .where(Expense.expense_date.between(prev_from, prev_to))
            .group_by(key)
        )
    else:
        if dimension == "product":
            key = Product.name
            join = (Product.__table__, Product.id == SalesTransaction.product_id)
        elif dimension == "category":
            key = Product.category
            join = (Product.__table__, Product.id == SalesTransaction.product_id)
        else:
            key = getattr(SalesTransaction, dimension)
            join = None

        value_expr, _ = _metric_expr(metric)
        conditions = _sales_filters(filters)

        def build(d_from: date, d_to: date):
            stmt = (
                select(key.label("key"), value_expr.label("v"))
                .select_from(
                    SalesTransaction.__table__.join(*join) if join else SalesTransaction.__table__
                )
                .where(and_(SalesTransaction.txn_date.between(d_from, d_to), *conditions))
                .group_by(key)
            )
            return stmt

        current_q = build(date_from, date_to)
        previous_q = build(prev_from, prev_to)

    current_rows = {r.key or "(unknown)": float(r.v) for r in (await db.execute(current_q)).all()}
    previous_rows = {r.key or "(unknown)": float(r.v) for r in (await db.execute(previous_q)).all()}

    keys = set(current_rows) | set(previous_rows)
    return [(k, current_rows.get(k, 0.0), previous_rows.get(k, 0.0)) for k in keys]


def _metric_expr(metric: str) -> tuple:
    if metric == "orders":
        return func.count(SalesTransaction.id), []
    if metric == "avg_order_value":
        # sum/count in one aggregate expression — safe in GROUP BY contexts
        return (
            func.sum(SalesTransaction.total_amount) / func.count(SalesTransaction.id),
            [],
        )
    if metric == "gross_margin":
        return (
            func.sum(
                SalesTransaction.total_amount
                - func.coalesce(Product.unit_cost, 0) * SalesTransaction.quantity
            ),
            [(Product.__table__, Product.id == SalesTransaction.product_id)],
        )
    return func.sum(SalesTransaction.total_amount), []


def _sales_filters(filters: dict[str, str | None]) -> list:
    conditions = []
    if filters.get("region"):
        conditions.append(SalesTransaction.region == filters["region"])
    if filters.get("channel"):
        conditions.append(SalesTransaction.channel == filters["channel"])
    if filters.get("category"):
        conditions.append(
            SalesTransaction.product_id.in_(
                select(Product.id).where(Product.category == filters["category"])
            )
        )
    return conditions