from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel

Granularity = Literal["day", "week", "month", "quarter", "year"]


class KpiCard(BaseModel):
    metric: str
    label: str | None = None
    unit: str | None = None
    value: float
    previous_value: float | None
    change_pct: float | None  # vs comparable previous period; None when no baseline
    target_value: float | None = None
    achievement_pct: float | None = None  # value/target when a target is defined
    status: str | None = None  # on_track | near_target | off_target (higher_is_better aware)


class KpiSummary(BaseModel):
    period_start: date
    period_end: date
    cards: list[KpiCard]


class KpiDefinitionOut(BaseModel):
    metric: str
    label: str
    formula: str
    unit: str
    higher_is_better: bool
    target_value: float | None
    threshold_low: float | None
    visibility: list[str]
    is_active: bool


class KpiDefinitionUpdate(BaseModel):
    label: str | None = None
    target_value: float | None = None
    threshold_low: float | None = None
    higher_is_better: bool | None = None
    unit: str | None = None
    visibility: list[str] | None = None
    is_active: bool | None = None


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
