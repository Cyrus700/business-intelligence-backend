"""Natural-language date parsing for the assistant.

Without this, every question collapses onto the same rolling window and the
assistant answers "revenue on 10 June" and "revenue on 1 Jan 2019" with the
same text — the "it always says the same thing" failure mode.
"""

from datetime import date, timedelta

import pytest

from app.core.clock import business_today
from app.services.ai.dates import parse_period
from app.services.ai.local_engine import _named_periods

TODAY = date(2026, 8, 9)  # a Sunday


@pytest.mark.parametrize(
    ("question", "start", "end"),
    [
        ("What was revenue on 2026-06-10?", date(2026, 6, 10), date(2026, 6, 10)),
        ("revenue yesterday", date(2026, 8, 8), date(2026, 8, 8)),
        ("sales today", date(2026, 8, 9), date(2026, 8, 9)),
        ("last 7 days", date(2026, 8, 3), date(2026, 8, 9)),
        ("past 3 months", date(2026, 5, 12), date(2026, 8, 9)),
        ("last month", date(2026, 7, 1), date(2026, 7, 31)),
        ("this month", date(2026, 8, 1), date(2026, 8, 9)),
        ("last year", date(2025, 1, 1), date(2025, 12, 31)),
        ("this year", date(2026, 1, 1), date(2026, 8, 9)),
        ("last quarter", date(2026, 4, 1), date(2026, 6, 30)),
        ("10 June 2026", date(2026, 6, 10), date(2026, 6, 10)),
        ("June 10", date(2026, 6, 10), date(2026, 6, 10)),
        ("12th December 2025", date(2025, 12, 12), date(2025, 12, 12)),
        ("revenue in June", date(2026, 6, 1), date(2026, 6, 30)),
        ("what about Dec 2025?", date(2025, 12, 1), date(2025, 12, 31)),
        ("in 2025", date(2025, 1, 1), date(2025, 12, 31)),
        # two ISO dates read as an inclusive span
        ("between 2026-06-10 and 2026-06-12", date(2026, 6, 10), date(2026, 6, 12)),
    ],
)
def test_parses_period(question, start, end):
    parsed = parse_period(question, TODAY)
    assert parsed is not None, question
    assert (parsed.start, parsed.end) == (start, end)


@pytest.mark.parametrize(
    "question",
    ["how are we doing?", "top products", "show me the forecast", "why did margin move?"],
)
def test_no_period_mentioned_returns_none(question):
    """Silence matters: the caller must be free to use its own default window."""
    assert parse_period(question, TODAY) is None


def test_bare_future_month_resolves_to_the_last_one_that_happened():
    # asked in August, "December" means last December, not a forecast
    parsed = parse_period("how was December?", TODAY)
    assert parsed is not None
    assert parsed.start == date(2025, 12, 1)


def test_reversed_iso_range_is_ordered():
    parsed = parse_period("from 2026-06-12 back to 2026-06-10", TODAY)
    assert parsed is not None
    assert (parsed.start, parsed.end) == (date(2026, 6, 10), date(2026, 6, 12))


def test_single_day_flag():
    assert parse_period("2026-06-10", TODAY).is_single_day
    assert not parse_period("last 7 days", TODAY).is_single_day


def test_invalid_calendar_date_is_rejected():
    assert parse_period("2026-02-30", TODAY) is None


# ── comparison splitting ───────────────────────────────────────────────────


def test_comparison_splits_into_two_distinct_periods():
    pair = _named_periods("Compare 10 June and 12 June 2026 revenue.")
    assert pair is not None
    assert [p.start for p in pair] == [date(2026, 6, 10), date(2026, 6, 12)]


def test_comparison_handles_vs_connective():
    # _named_periods parses against the real business date, so the expectation
    # is derived from today rather than pinned to a month that goes stale.
    pair = _named_periods("last month vs this month")
    assert pair is not None
    this_month = business_today().replace(day=1)
    last_month = (this_month - timedelta(days=1)).replace(day=1)
    assert pair[0].start == last_month
    assert pair[1].start == this_month


def test_single_period_question_is_not_a_comparison():
    assert _named_periods("revenue this month") is None
