"""The business clock.

Every "today", "now" and date-window default in the system resolves through
this module so that one calendar day means the same thing in the dashboard
filters, the SQL windows, the ETL validator and the AI assistant's answers.

The anchor is a fixed zone (``Asia/Kathmandu``) rather than the server's local
time or UTC: the warehouse is a Nepali retail business, so a sale rung up at
21:00 in Kathmandu belongs to that Nepali day. Under UTC it would land on the
previous day (UTC+05:45), which is exactly the off-by-one that makes a "today"
filter disagree with what the shop floor saw.

Never call ``date.today()`` / ``datetime.now()`` directly in application code —
they follow the host's TZ env var and silently differ between a laptop, CI and
the deployed container.
"""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

BUSINESS_TZ = ZoneInfo("Asia/Kathmandu")
BUSINESS_TZ_NAME = "Asia/Kathmandu"


def business_now() -> datetime:
    """Current instant as a timezone-aware datetime in the business zone."""
    return datetime.now(BUSINESS_TZ)


def business_today() -> date:
    """The current business day."""
    return business_now().date()


def business_yesterday() -> date:
    return business_today() - timedelta(days=1)


def to_business_date(moment: datetime) -> date:
    """Calendar day a timestamp falls on, in the business zone.

    Naive datetimes are assumed to already be in the business zone — that is
    how the warehouse's timestamp columns are written.
    """
    if moment.tzinfo is None:
        return moment.date()
    return moment.astimezone(BUSINESS_TZ).date()


def day_bounds(day: date) -> tuple[datetime, datetime]:
    """Half-open [start, end) instants covering ``day`` in the business zone.

    Use for filtering timestamp columns; ``BETWEEN`` on a timestamp would drop
    rows recorded after 00:00:00.000 on the final day.
    """
    start = datetime.combine(day, datetime.min.time(), tzinfo=BUSINESS_TZ)
    return start, start + timedelta(days=1)
