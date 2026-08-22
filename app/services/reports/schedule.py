"""Next-run-date math for recurring report schedules.

Kept separate from the API layer and the worker so both compute "when does
this fire next" the same way — the API uses it once at create/update time,
the worker uses it again every time a schedule fires to roll it forward.
"""

from datetime import date, timedelta


def compute_next_run(
    frequency: str,
    day_of_week: int | None,
    day_of_month: int | None,
    today: date,
    *,
    include_today: bool = True,
) -> date:
    """Next business date this schedule fires on, on/after ``today``.

    ``include_today=False`` forces the result strictly after ``today`` — used
    when advancing a schedule right after it just fired, so it can't fire
    twice on the same day.
    """
    if frequency == "weekly":
        assert day_of_week is not None
        delta = (day_of_week - today.weekday()) % 7
        if delta == 0 and not include_today:
            delta = 7
        return today + timedelta(days=delta)

    if frequency == "monthly":
        assert day_of_month is not None
        candidate = _same_month_day(today, day_of_month)
        if candidate < today or (candidate == today and not include_today):
            candidate = _same_month_day(_first_of_next_month(today), day_of_month)
        return candidate

    raise ValueError(f"unknown frequency: {frequency}")


def _first_of_next_month(d: date) -> date:
    if d.month == 12:
        return date(d.year + 1, 1, 1)
    return date(d.year, d.month + 1, 1)


def _same_month_day(d: date, day_of_month: int) -> date:
    # day_of_month is capped at 28 by the schema, so it always exists in
    # every month — no need to clamp against short/leap Februarys.
    return date(d.year, d.month, day_of_month)
