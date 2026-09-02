"""Driver decomposition, the revenue bridge and concentration.

These are the arithmetic the assistant is forbidden from doing in its head, so
they have to be right here or the ban buys nothing.
"""

import pytest

from app.services.ml.diagnostics import concentration, decompose, price_volume_bridge


def test_decompose_ranks_drivers_and_drags():
    current = {"Wai Wai": 300.0, "Chau Chau": 500.0, "Rara": 100.0}
    previous = {"Wai Wai": 500.0, "Chau Chau": 300.0, "Rara": 100.0}
    b = decompose(current, previous)

    assert b.total_delta == 0.0
    assert [c.key for c in b.drivers] == ["Chau Chau"]
    assert [c.key for c in b.drags] == ["Wai Wai"]
    # Rara did not move, so it is not part of the explanation at all.
    assert "Rara" not in {c.key for c in b.drivers + b.drags}


def test_contribution_is_measured_against_gross_movement():
    """Netting first gives '400% of the change' whenever gains nearly cancel."""
    b = decompose({"a": 300.0, "b": 100.0}, {"a": 100.0, "b": 300.0})
    assert b.total_delta == 0.0
    contributions = {c.key: c.contribution_pct for c in b.drivers + b.drags}
    assert contributions["a"] == pytest.approx(50.0)
    assert contributions["b"] == pytest.approx(-50.0)


def test_decompose_flags_new_and_lost_members():
    b = decompose({"new item": 400.0, "kept": 100.0}, {"gone": 400.0, "kept": 100.0})
    assert b.new_members == ["new item"]
    assert b.lost_members == ["gone"]


def test_decompose_reports_change_pct_and_survives_a_zero_base():
    grew = decompose({"a": 150.0}, {"a": 100.0})
    assert grew.change_pct == 50.0
    from_nothing = decompose({"a": 150.0}, {})
    assert from_nothing.change_pct is None  # no base to divide by, not "infinite growth"


def test_bridge_terms_reconstruct_the_movement_exactly():
    """volume + value + interaction must equal the revenue delta, always."""
    b = price_volume_bridge(
        orders_current=120,
        orders_previous=100,
        revenue_current=132_000,
        revenue_previous=100_000,
    )
    assert b.volume_effect + b.value_effect + b.interaction_effect == pytest.approx(b.revenue_delta)


def test_bridge_names_a_pure_volume_move():
    b = price_volume_bridge(
        orders_current=200,
        orders_previous=100,
        revenue_current=200_000,
        revenue_previous=100_000,
    )
    assert b.aov_current == b.aov_previous
    assert b.value_effect == 0
    assert b.verdict == "driven by order volume"


def test_bridge_names_a_pure_value_move():
    b = price_volume_bridge(
        orders_current=100,
        orders_previous=100,
        revenue_current=150_000,
        revenue_previous=100_000,
    )
    assert b.volume_effect == 0
    assert b.verdict == "driven by order value"


def test_bridge_survives_a_period_with_no_orders():
    b = price_volume_bridge(0, 0, 0.0, 0.0)
    assert b.aov_current == 0
    assert b.verdict == "no material change"


def test_concentration_flags_a_dominant_member():
    c = concentration({"a": 900.0, "b": 50.0, "c": 50.0})
    assert c.top1_share_pct == pytest.approx(90.0)
    assert c.hhi > 0.25
    assert c.risk.startswith("high")
    assert c.leaders[0] == "a"


def test_concentration_calls_an_even_spread_low_risk():
    c = concentration({k: 100.0 for k in "abcdefghij"})
    assert c.hhi == pytest.approx(0.1)
    assert c.risk.startswith("low")


def test_concentration_on_no_data():
    c = concentration({})
    assert c.members == 0
    assert c.risk == "no data"
