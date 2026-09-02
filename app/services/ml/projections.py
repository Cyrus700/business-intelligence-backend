"""Forward-looking analysis: where the period lands, and what would change it.

The trained forecaster answers "what will the next 30 days look like". Two
questions users actually ask are neither that nor a KPI lookup:

  * "Are we going to hit the month?" — a projection of the *current, partly
    elapsed* period that combines what already happened with what the remaining
    days are likely to add.
  * "What if we lifted order value 5%?" — a scenario, where the point is the
    delta against the baseline, not the absolute number.

Both are computed here rather than left to the model, because both are
arithmetic over live figures and a language model doing arithmetic is the
failure this codebase spends most of its effort preventing.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.clock import business_today
from app.services.analytics.queries import Filters, kpi_summary, kpi_timeseries

#: Weekday profiles need a few observations each before they beat a flat mean.
MIN_DAYS_FOR_WEEKDAY_PROFILE = 14
#: z for a 95% interval, applied to the accumulated variance of the days left.
Z_95 = 1.96


@dataclass
class PeriodProjection:
    metric: str
    period_label: str
    period_start: date
    period_end: date
    days_elapsed: int
    days_remaining: int
    actual_to_date: float
    projected_remainder: float
    projected_total: float
    lower_bound: float
    upper_bound: float
    #: How the projection was built, so the answer can say it out loud instead
    #: of presenting a run-rate as if it were a trained model.
    method: str
    daily_run_rate: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "period": self.period_label,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "days_elapsed": self.days_elapsed,
            "days_remaining": self.days_remaining,
            "actual_to_date": round(self.actual_to_date, 2),
            "projected_remainder": round(self.projected_remainder, 2),
            "projected_total": round(self.projected_total, 2),
            "lower_bound": round(self.lower_bound, 2),
            "upper_bound": round(self.upper_bound, 2),
            "daily_run_rate": round(self.daily_run_rate, 2),
            "method": self.method,
        }


def _weekday_profile(history: list[tuple[date, float]]) -> dict[int, float]:
    """Mean value per weekday. Trade is not flat across the week.

    Projecting a month that has four weekends left off a flat daily mean is
    wrong in a direction that depends entirely on which days remain, which is
    why the error looks random rather than like a bias.
    """
    buckets: dict[int, list[float]] = {}
    for day, value in history:
        buckets.setdefault(day.weekday(), []).append(value)
    return {wd: statistics.fmean(vals) for wd, vals in buckets.items() if vals}


def project_period(
    history: list[tuple[date, float]],
    period_start: date,
    period_end: date,
    today: date,
    metric: str = "revenue",
    period_label: str = "",
) -> PeriodProjection:
    """Project a partly-elapsed period from its own days plus recent history.

    ``history`` is daily observations, and may run before ``period_start`` —
    the extra days only feed the weekday profile and the variance band, never
    the actual-to-date total.
    """
    in_period = [(d, v) for d, v in history if period_start <= d <= min(today, period_end)]
    actual = sum(v for _, v in in_period)
    days_elapsed = len(in_period)
    remaining_days = [
        period_start + timedelta(days=i)
        for i in range((period_end - period_start).days + 1)
        if period_start + timedelta(days=i) > today
    ]

    if not history or days_elapsed == 0:
        return PeriodProjection(
            metric=metric,
            period_label=period_label or f"{period_start} → {period_end}",
            period_start=period_start,
            period_end=period_end,
            days_elapsed=0,
            days_remaining=len(remaining_days),
            actual_to_date=0.0,
            projected_remainder=0.0,
            projected_total=0.0,
            lower_bound=0.0,
            upper_bound=0.0,
            method="no data for this period yet",
            daily_run_rate=0.0,
        )

    values = [v for _, v in history]
    flat_mean = statistics.fmean(values)
    profile = _weekday_profile(history)
    use_profile = len(history) >= MIN_DAYS_FOR_WEEKDAY_PROFILE and len(profile) >= 5

    if use_profile:
        remainder = sum(profile.get(d.weekday(), flat_mean) for d in remaining_days)
        method = f"weekday-adjusted run rate over {len(history)} days of history"
    else:
        remainder = flat_mean * len(remaining_days)
        method = f"flat run rate over {len(history)} days of history"

    # The band widens with the square root of the days left, not linearly:
    # independent daily errors partly cancel, and a linear band on a 30-day
    # horizon is so wide it says nothing.
    spread = statistics.pstdev(values) if len(values) > 1 else 0.0
    margin = Z_95 * spread * (len(remaining_days) ** 0.5)
    total = actual + remainder

    return PeriodProjection(
        metric=metric,
        period_label=period_label or f"{period_start} → {period_end}",
        period_start=period_start,
        period_end=period_end,
        days_elapsed=days_elapsed,
        days_remaining=len(remaining_days),
        actual_to_date=actual,
        projected_remainder=remainder,
        projected_total=total,
        lower_bound=max(0.0, total - margin),
        upper_bound=total + margin,
        method=method,
        daily_run_rate=actual / days_elapsed,
    )


@dataclass
class Scenario:
    """A what-if against the live baseline.

    Revenue is modelled as orders × average order value, so a scenario moves
    one or both and the interaction falls out of the multiplication rather than
    being approximated by adding the two percentages.
    """

    baseline_revenue: float
    baseline_orders: float
    baseline_aov: float
    baseline_expenses: float
    baseline_profit: float
    scenario_revenue: float
    scenario_orders: float
    scenario_aov: float
    scenario_expenses: float
    scenario_profit: float
    assumptions: dict[str, float]

    @property
    def revenue_delta(self) -> float:
        return self.scenario_revenue - self.baseline_revenue

    @property
    def profit_delta(self) -> float:
        return self.scenario_profit - self.baseline_profit

    def as_dict(self) -> dict[str, Any]:
        return {
            "assumptions": self.assumptions,
            "baseline": {
                "revenue": round(self.baseline_revenue, 2),
                "orders": round(self.baseline_orders, 2),
                "avg_order_value": round(self.baseline_aov, 2),
                "expenses": round(self.baseline_expenses, 2),
                "profit": round(self.baseline_profit, 2),
            },
            "scenario": {
                "revenue": round(self.scenario_revenue, 2),
                "orders": round(self.scenario_orders, 2),
                "avg_order_value": round(self.scenario_aov, 2),
                "expenses": round(self.scenario_expenses, 2),
                "profit": round(self.scenario_profit, 2),
            },
            "delta": {
                "revenue": round(self.revenue_delta, 2),
                "profit": round(self.profit_delta, 2),
                "revenue_pct": (
                    round(self.revenue_delta / self.baseline_revenue * 100, 1)
                    if self.baseline_revenue
                    else None
                ),
            },
        }


def simulate(
    revenue: float,
    orders: float,
    expenses: float,
    *,
    orders_change_pct: float = 0.0,
    aov_change_pct: float = 0.0,
    expense_change_pct: float = 0.0,
) -> Scenario:
    aov = revenue / orders if orders else 0.0
    new_orders = orders * (1 + orders_change_pct / 100)
    new_aov = aov * (1 + aov_change_pct / 100)
    new_revenue = new_orders * new_aov
    new_expenses = expenses * (1 + expense_change_pct / 100)
    return Scenario(
        baseline_revenue=revenue,
        baseline_orders=orders,
        baseline_aov=aov,
        baseline_expenses=expenses,
        baseline_profit=revenue - expenses,
        scenario_revenue=new_revenue,
        scenario_orders=new_orders,
        scenario_aov=new_aov,
        scenario_expenses=new_expenses,
        scenario_profit=new_revenue - new_expenses,
        assumptions={
            "orders_change_pct": orders_change_pct,
            "aov_change_pct": aov_change_pct,
            "expense_change_pct": expense_change_pct,
        },
    )


# ── warehouse-backed wrappers ──────────────────────────────────────────────

def month_bounds(day: date) -> tuple[date, date]:
    start = day.replace(day=1)
    next_month = (start + timedelta(days=32)).replace(day=1)
    return start, next_month - timedelta(days=1)


def quarter_bounds(day: date) -> tuple[date, date]:
    first_month = 3 * ((day.month - 1) // 3) + 1
    start = date(day.year, first_month, 1)
    next_q = (start + timedelta(days=100)).replace(day=1)
    return start, next_q - timedelta(days=1)


async def project_current_period(
    db: AsyncSession,
    metric: str = "revenue",
    period: str = "month",
    lookback_days: int = 90,
    org_id=None,
) -> PeriodProjection:
    """Project how the current month or quarter lands, from live daily data."""
    today = business_today()
    start, end = quarter_bounds(today) if period == "quarter" else month_bounds(today)
    label = f"Q{(start.month - 1) // 3 + 1} {start.year}" if period == "quarter" else (
        start.strftime("%B %Y")
    )

    history_from = min(start, today - timedelta(days=lookback_days - 1))
    points = await kpi_timeseries(
        db, Filters(date_from=history_from, date_to=today, org_id=org_id), metric, "day"
    )
    history = [(p["period"], float(p["value"])) for p in points]
    return project_period(history, start, end, today, metric=metric, period_label=label)


async def simulate_current_period(
    db: AsyncSession,
    start: date,
    end: date,
    *,
    orders_change_pct: float = 0.0,
    aov_change_pct: float = 0.0,
    expense_change_pct: float = 0.0,
    org_id=None,
) -> Scenario:
    cards = {c["metric"]: c for c in await kpi_summary(db, Filters(date_from=start, date_to=end, org_id=org_id))}
    return simulate(
        revenue=float(cards.get("revenue", {}).get("value") or 0.0),
        orders=float(cards.get("orders", {}).get("value") or 0.0),
        expenses=float(cards.get("expense_total", {}).get("value") or 0.0),
        orders_change_pct=orders_change_pct,
        aov_change_pct=aov_change_pct,
        expense_change_pct=expense_change_pct,
    )
