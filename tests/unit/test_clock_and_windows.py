"""The business clock and the assistant's date-window resolution.

These guard the two places a date can silently drift: the timezone "today" is
computed in, and the translation from a user's phrasing ("yesterday", "last
month") into an inclusive SQL window.
"""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from app.core.clock import (
    BUSINESS_TZ,
    BUSINESS_TZ_NAME,
    business_now,
    business_today,
    business_yesterday,
    day_bounds,
    to_business_date,
)
from app.services.ai.tools import _window


def test_business_zone_is_nepal():
    assert BUSINESS_TZ_NAME == "Asia/Kathmandu"
    assert business_now().tzinfo is BUSINESS_TZ


def test_business_today_matches_zone_not_utc():
    assert business_today() == datetime.now(BUSINESS_TZ).date()


def test_late_evening_in_kathmandu_is_still_the_same_business_day():
    """22:00 in Kathmandu is 16:15 UTC — same day. 00:30 is the previous UTC day.

    This is the off-by-one that a naive utcnow().date() introduces: a sale rung
    up just after midnight locally would be filed on the previous business day.
    """
    just_after_midnight = datetime(2026, 6, 10, 0, 30, tzinfo=BUSINESS_TZ)
    assert to_business_date(just_after_midnight) == date(2026, 6, 10)
    # the same instant expressed in UTC is still the 9th by UTC's calendar
    assert just_after_midnight.astimezone(ZoneInfo("UTC")).date() == date(2026, 6, 9)


def test_yesterday_is_one_day_back():
    assert business_yesterday() == business_today() - timedelta(days=1)


def test_day_bounds_are_half_open_and_cover_the_whole_day():
    start, end = day_bounds(date(2026, 6, 10))
    assert start == datetime(2026, 6, 10, 0, 0, tzinfo=BUSINESS_TZ)
    assert end == datetime(2026, 6, 11, 0, 0, tzinfo=BUSINESS_TZ)
    # a 23:59 timestamp must fall inside the window
    assert start <= datetime(2026, 6, 10, 23, 59, tzinfo=BUSINESS_TZ) < end


# ── assistant window resolution ────────────────────────────────────────────

def test_default_window_is_trailing_30_days_inclusive():
    date_from, date_to = _window({})
    assert date_to == business_today()
    assert (date_to - date_from).days == 29  # 30 inclusive days


def test_period_today_is_a_single_day():
    assert _window({"period": "today"}) == (business_today(), business_today())


def test_period_yesterday_is_a_single_day():
    y = business_yesterday()
    assert _window({"period": "yesterday"}) == (y, y)


def test_period_last_7_days_is_inclusive():
    date_from, date_to = _window({"period": "last_7_days"})
    assert (date_to - date_from).days == 6
    assert date_to == business_today()


def test_period_last_month_is_the_whole_previous_calendar_month():
    date_from, date_to = _window({"period": "last_month"})
    assert date_from.day == 1
    # the end must be the last day of that same month
    assert (date_to + timedelta(days=1)).day == 1
    assert date_to.month == date_from.month


def test_single_date_argument_pins_one_day():
    assert _window({"date": "2026-06-12"}) == (date(2026, 6, 12), date(2026, 6, 12))


def test_relative_word_accepted_as_a_date_value():
    assert _window({"date": "yesterday"}) == (business_yesterday(), business_yesterday())


def test_reversed_range_is_swapped_not_dropped():
    assert _window({"date_from": "2026-06-30", "date_to": "2026-06-01"}) == (
        date(2026, 6, 1),
        date(2026, 6, 30),
    )


def test_explicit_dates_beat_period():
    assert _window({"date_from": "2026-01-01", "date_to": "2026-01-31", "period": "today"}) == (
        date(2026, 1, 1),
        date(2026, 1, 31),
    )


def test_unparseable_date_falls_back_instead_of_raising():
    date_from, date_to = _window({"date_from": "not-a-date", "date_to": "2026-06-30"})
    assert date_to == date(2026, 6, 30)
    assert date_from == date(2026, 6, 30) - timedelta(days=29)
