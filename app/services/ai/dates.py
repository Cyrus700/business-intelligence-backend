"""Natural-language date extraction for the assistant.

The LLM resolves dates by calling tools with explicit arguments. The local
engine has no such luxury: it sees only the raw question, so without this it
answers every question for the same default 30-day window — which is how
"revenue on 10 June" and "revenue on 1 Jan 2019" end up with identical replies.

Returns an inclusive [start, end] window on the business calendar plus a label
to quote back, so the answer always states the period it actually covers.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta

from app.core.clock import business_today

MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))


@dataclass(frozen=True)
class ParsedPeriod:
    start: date
    end: date
    label: str
    #: True when the user named an exact day/month, so an empty result must be
    #: reported as "no data for that date" rather than quietly widened.
    explicit: bool = True

    @property
    def is_single_day(self) -> bool:
        return self.start == self.end


def _month_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def _fmt(d: date) -> str:
    return d.strftime("%-d %b %Y") if hasattr(d, "strftime") else d.isoformat()


def _day(d: date) -> ParsedPeriod:
    return ParsedPeriod(d, d, _fmt(d))


def _span(start: date, end: date, label: str) -> ParsedPeriod:
    return ParsedPeriod(start, end, label)


def parse_period(question: str, today: date | None = None) -> ParsedPeriod | None:
    """Best-effort window for a question. None means "no period mentioned".

    Deliberately conservative: an unrecognised phrase returns None so the caller
    falls back to its default window rather than answering about a period the
    user never asked for.
    """
    today = today or business_today()
    q = question.lower().strip()

    # ── explicit ISO date or ISO range ────────────────────────────────
    iso = re.findall(r"\b(\d{4})-(\d{2})-(\d{2})\b", q)
    if len(iso) >= 2:
        first, second = (date(int(y), int(m), int(d)) for y, m, d in iso[:2])
        start, end = sorted((first, second))
        return _span(start, end, f"{_fmt(start)} → {_fmt(end)}")
    if len(iso) == 1:
        y, m, d = iso[0]
        try:
            return _day(date(int(y), int(m), int(d)))
        except ValueError:
            return None

    # ── relative days ─────────────────────────────────────────────────
    if re.search(r"\btoday\b|\bso far today\b", q):
        return ParsedPeriod(today, today, "today")
    if re.search(r"\byesterday\b", q):
        y = today - timedelta(days=1)
        return ParsedPeriod(y, y, "yesterday")
    if re.search(r"\bday before yesterday\b", q):
        y = today - timedelta(days=2)
        return ParsedPeriod(y, y, _fmt(y))

    # ── rolling windows: "last 7 days", "past 3 months" ───────────────
    rolling = re.search(r"\b(?:last|past|previous)\s+(\d{1,3})\s*(day|week|month|year)s?\b", q)
    if rolling:
        n = int(rolling.group(1))
        unit = rolling.group(2)
        days = {"day": 1, "week": 7, "month": 30, "year": 365}[unit] * n
        start = today - timedelta(days=days - 1)
        return _span(start, today, f"last {n} {unit}{'s' if n != 1 else ''}")

    # ── named calendar periods ────────────────────────────────────────
    if re.search(r"\b(last|previous)\s+week\b", q):
        end = today - timedelta(days=today.weekday() + 1)
        return _span(end - timedelta(days=6), end, "last week")
    if re.search(r"\bthis\s+week\b", q):
        return _span(today - timedelta(days=today.weekday()), today, "this week")
    if re.search(r"\b(last|previous)\s+month\b", q):
        end = today.replace(day=1) - timedelta(days=1)
        return _span(end.replace(day=1), end, end.strftime("%B %Y"))
    if re.search(r"\bthis\s+month\b", q):
        return _span(today.replace(day=1), today, today.strftime("%B %Y"))
    if re.search(r"\b(last|previous)\s+year\b", q):
        year = today.year - 1
        return _span(date(year, 1, 1), date(year, 12, 31), str(year))
    if re.search(r"\bthis\s+year\b|\bytd\b|\byear to date\b", q):
        return _span(date(today.year, 1, 1), today, f"{today.year} to date")
    if re.search(r"\b(last|previous)\s+quarter\b", q):
        q_start_month = 3 * ((today.month - 1) // 3) + 1
        this_q_start = date(today.year, q_start_month, 1)
        end = this_q_start - timedelta(days=1)
        start = date(end.year, 3 * ((end.month - 1) // 3) + 1, 1)
        return _span(start, end, f"Q{(start.month - 1) // 3 + 1} {start.year}")

    # ── "10 June 2026" / "June 10" / "10th June" ──────────────────────
    dm = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTH_RE})\b\.?\s*(\d{{4}})?", q)
    if dm:
        day_n, month_name, year_s = dm.groups()
        return _resolve_day(int(day_n), MONTHS[month_name], year_s, today)

    md = re.search(rf"\b({MONTH_RE})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b,?\s*(\d{{4}})?", q)
    if md:
        month_name, day_n, year_s = md.groups()
        return _resolve_day(int(day_n), MONTHS[month_name], year_s, today)

    # ── bare month, optionally with a year: "in June", "June 2026" ────
    bare = re.search(rf"\b({MONTH_RE})\b\.?\s*(\d{{4}})?", q)
    if bare:
        month_name, year_s = bare.groups()
        month = MONTHS[month_name]
        year = int(year_s) if year_s else today.year
        # a bare future month almost always means the one just gone
        if not year_s and month > today.month:
            year -= 1
        start = date(year, month, 1)
        return _span(start, _month_end(year, month), start.strftime("%B %Y"))

    # ── bare year: "in 2025" ──────────────────────────────────────────
    year_only = re.search(r"\b(20\d{2})\b", q)
    if year_only:
        year = int(year_only.group(1))
        return _span(date(year, 1, 1), date(year, 12, 31), str(year))

    return None


def _resolve_day(day_n: int, month: int, year_s: str | None, today: date) -> ParsedPeriod | None:
    year = int(year_s) if year_s else today.year
    try:
        resolved = date(year, month, day_n)
    except ValueError:
        return None
    # An unqualified date in the future means last year's, not a prediction.
    if not year_s and resolved > today:
        try:
            resolved = date(year - 1, month, day_n)
        except ValueError:
            return None
    return _day(resolved)
