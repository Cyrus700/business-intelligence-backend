"""Period-end projection and what-if scenarios."""

from datetime import date, timedelta

import pytest

from app.services.ml.projections import (
    month_bounds,
    project_period,
    quarter_bounds,
    simulate,
)


def _flat_history(start: date, days: int, value: float) -> list[tuple[date, float]]:
    return [(start + timedelta(days=i), value) for i in range(days)]


def test_projection_adds_the_remaining_days_to_the_actual():
    start, end = date(2026, 6, 1), date(2026, 6, 30)
    today = date(2026, 6, 10)  # 10 days in, 20 to go
    p = project_period(_flat_history(start, 10, 1000.0), start, end, today, period_label="June")

    assert p.days_elapsed == 10
    assert p.days_remaining == 20
    assert p.actual_to_date == pytest.approx(10_000)
    assert p.projected_total == pytest.approx(30_000)
    assert p.daily_run_rate == pytest.approx(1000)


def test_a_flat_series_projects_with_a_zero_width_band():
    """No observed variance means no honest reason to widen the range."""
    start, end = date(2026, 6, 1), date(2026, 6, 30)
    p = project_period(_flat_history(start, 10, 1000.0), start, end, date(2026, 6, 10))
    assert p.lower_bound == pytest.approx(p.upper_bound)


def test_a_volatile_series_widens_the_band():
    start, end = date(2026, 6, 1), date(2026, 6, 30)
    history = [(start + timedelta(days=i), 1000.0 if i % 2 else 200.0) for i in range(10)]
    p = project_period(history, start, end, date(2026, 6, 10))
    assert p.upper_bound > p.projected_total > p.lower_bound
    assert p.lower_bound >= 0  # revenue cannot be projected negative


def test_weekday_profile_kicks_in_with_enough_history():
    """Weekends trade differently; a flat mean misprices whichever days remain."""
    start, end = date(2026, 6, 1), date(2026, 6, 30)
    history = [
        (start + timedelta(days=i), 200.0 if (start + timedelta(days=i)).weekday() >= 5 else 1000.0) for i in range(21)
    ]
    p = project_period(history, start, end, start + timedelta(days=20))
    assert "weekday-adjusted" in p.method


def test_short_history_falls_back_to_a_flat_run_rate():
    start, end = date(2026, 6, 1), date(2026, 6, 30)
    p = project_period(_flat_history(start, 5, 1000.0), start, end, date(2026, 6, 5))
    assert "flat run rate" in p.method


def test_a_period_with_no_data_projects_nothing():
    start, end = date(2026, 6, 1), date(2026, 6, 30)
    p = project_period([], start, end, date(2026, 6, 10))
    assert p.projected_total == 0
    assert p.method == "no data for this period yet"


def test_a_finished_period_has_nothing_left_to_project():
    start, end = date(2026, 6, 1), date(2026, 6, 30)
    p = project_period(_flat_history(start, 30, 1000.0), start, end, date(2026, 7, 5))
    assert p.days_remaining == 0
    assert p.projected_total == pytest.approx(p.actual_to_date)


def test_scenario_compounds_rather_than_adding_percentages():
    """+10% orders and +10% value is +21% revenue, not +20%."""
    sc = simulate(
        revenue=100_000,
        orders=100,
        expenses=40_000,
        orders_change_pct=10,
        aov_change_pct=10,
    )
    assert sc.scenario_revenue == pytest.approx(121_000)
    assert sc.revenue_delta == pytest.approx(21_000)


def test_scenario_moves_profit_with_expenses():
    sc = simulate(revenue=100_000, orders=100, expenses=40_000, expense_change_pct=-25)
    assert sc.baseline_profit == pytest.approx(60_000)
    assert sc.scenario_profit == pytest.approx(70_000)
    assert sc.profit_delta == pytest.approx(10_000)


def test_scenario_survives_a_baseline_with_no_orders():
    sc = simulate(revenue=0, orders=0, expenses=1_000, orders_change_pct=50)
    assert sc.baseline_aov == 0
    assert sc.scenario_revenue == 0
    assert sc.as_dict()["delta"]["revenue_pct"] is None


def test_month_and_quarter_bounds():
    assert month_bounds(date(2026, 2, 14)) == (date(2026, 2, 1), date(2026, 2, 28))
    assert month_bounds(date(2026, 12, 31)) == (date(2026, 12, 1), date(2026, 12, 31))
    assert quarter_bounds(date(2026, 8, 18)) == (date(2026, 7, 1), date(2026, 9, 30))
    assert quarter_bounds(date(2026, 1, 1)) == (date(2026, 1, 1), date(2026, 3, 31))
