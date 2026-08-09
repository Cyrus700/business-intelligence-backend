from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel

Granularity = Literal["day", "week", "month"]


class KpiCard(BaseModel):
    metric: str
    value: float
    previous_value: float | None
    change_pct: float | None  # vs comparable previous period; None when no baseline


class KpiSummary(BaseModel):
    period_start: date
    period_end: date
    cards: list[KpiCard]


class TimeseriesPoint(BaseModel):
    period: date
    value: float


class Timeseries(BaseModel):
    metric: str
    granularity: Granularity
    points: list[TimeseriesPoint]


class DimensionRow(BaseModel):
    key: str  # product name / region / channel / category
    sku: str | None = None
    quantity: int | None = None
    orders: int
    revenue: float
    share_pct: float


class TransactionRow(BaseModel):
    id: int
    txn_date: date
    product: str | None
    sku: str | None
    customer: str | None
    channel: str | None
    region: str | None
    quantity: int
    unit_price: float | None = None
    discount: float | None = None
    total_amount: float | None = None
    redacted: bool | None = None  # field-level redaction for analyst role


class Paginated[T](BaseModel):
    items: list[T]
    total: int
    page: int
    page_size: int


class PnlRow(BaseModel):
    month: date
    revenue: float
    expenses: float
    gross_margin: float
    net: float


class InventoryRow(BaseModel):
    sku: str
    product: str
    category: str | None
    snapshot_date: date
    quantity_on_hand: int
    reorder_level: int
    below_reorder: bool
    warehouse: str | None


class TableCoverage(BaseModel):
    first_date: date | None
    last_date: date | None
    row_count: int
    last_ingested_at: datetime | None


class DataCoverage(BaseModel):
    """What the warehouse holds, so clients never mistake "not loaded" for zero."""

    sales: TableCoverage
    expenses: TableCoverage
    inventory: TableCoverage
    first_date: date | None
    last_date: date | None
    last_ingested_at: datetime | None
    today: date
    timezone: str
    days_behind: int | None
