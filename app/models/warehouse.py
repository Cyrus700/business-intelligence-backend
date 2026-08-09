import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import CheckConstraint, ForeignKey, Index, Numeric, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, uuid_pk

Money = Numeric(14, 2)


class Product(Base, TimestampMixin):
    __tablename__ = "products"

    id: Mapped[uuid.UUID] = uuid_pk()
    sku: Mapped[str] = mapped_column(unique=True)
    name: Mapped[str]
    category: Mapped[str | None]
    unit_cost: Mapped[Decimal | None] = mapped_column(Money)
    unit_price: Mapped[Decimal | None] = mapped_column(Money)


class Customer(Base, TimestampMixin):
    __tablename__ = "customers"

    id: Mapped[uuid.UUID] = uuid_pk()
    name: Mapped[str]
    segment: Mapped[str | None]
    city: Mapped[str | None]
    region: Mapped[str | None]

    __table_args__ = (
        CheckConstraint(
            "segment IS NULL OR segment IN ('retail', 'wholesale', 'online')",
            name="valid_segment",
        ),
    )


class SalesTransaction(Base):
    __tablename__ = "sales_transactions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    txn_date: Mapped[date]
    product_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("products.id", ondelete="SET NULL")
    )
    customer_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("customers.id", ondelete="SET NULL")
    )
    quantity: Mapped[int]
    unit_price: Mapped[Decimal] = mapped_column(Money)
    discount: Mapped[Decimal] = mapped_column(Money, default=0)
    total_amount: Mapped[Decimal] = mapped_column(Money)
    channel: Mapped[str | None]
    region: Mapped[str | None]
    # natural-key hash for idempotent loads (Phase 2)
    row_hash: Mapped[str | None] = mapped_column(unique=True)
    source_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("data_sources.id", ondelete="SET NULL")
    )
    etl_job_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("etl_jobs.id", ondelete="SET NULL")
    )
    # When this row landed in the warehouse, on the business clock. Distinct
    # from the business date above: a February sale uploaded in March has
    # txn_date=February, ingested_at=March. Powers "uploaded today" views and
    # lets the assistant tell "no sales that day" from "not loaded yet".
    ingested_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (
        Index("ix_sales_txn_date", "txn_date"),
        Index("ix_sales_ingested_at", "ingested_at"),
        Index("ix_sales_txn_date_product", "txn_date", "product_id"),
        Index("ix_sales_txn_date_region", "txn_date", "region"),
        CheckConstraint("quantity > 0", name="positive_quantity"),
    )


class Expense(Base):
    __tablename__ = "expenses"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    expense_date: Mapped[date]
    category: Mapped[str]
    amount: Mapped[Decimal] = mapped_column(Money)
    department: Mapped[str | None]
    description: Mapped[str | None]
    row_hash: Mapped[str | None] = mapped_column(unique=True)
    source_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("data_sources.id", ondelete="SET NULL")
    )
    etl_job_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("etl_jobs.id", ondelete="SET NULL")
    )
    # When this row landed in the warehouse, on the business clock. Distinct
    # from the business date above: a February sale uploaded in March has
    # txn_date=February, ingested_at=March. Powers "uploaded today" views and
    # lets the assistant tell "no sales that day" from "not loaded yet".
    ingested_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (
        Index("ix_expenses_expense_date", "expense_date"),
        Index("ix_expenses_ingested_at", "ingested_at"),
        CheckConstraint(
            "category IN ('rent', 'salaries', 'utilities', 'marketing', 'logistics', 'other')",
            name="valid_category",
        ),
        CheckConstraint("amount >= 0", name="non_negative_amount"),
    )


class InventoryLevel(Base):
    __tablename__ = "inventory_levels"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    snapshot_date: Mapped[date]
    product_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("products.id", ondelete="CASCADE"))
    quantity_on_hand: Mapped[int]
    reorder_level: Mapped[int] = mapped_column(default=0)
    warehouse: Mapped[str | None]
    source_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("data_sources.id", ondelete="SET NULL")
    )
    # When this row landed in the warehouse, on the business clock. Distinct
    # from the business date above: a February sale uploaded in March has
    # txn_date=February, ingested_at=March. Powers "uploaded today" views and
    # lets the assistant tell "no sales that day" from "not loaded yet".
    ingested_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (
        Index("ix_inventory_snapshot_date", "snapshot_date"),
        Index("ix_inventory_ingested_at", "ingested_at"),
        UniqueConstraint("snapshot_date", "product_id", "warehouse", name="uq_inventory_snapshot"),
    )


class KpiSnapshot(Base):
    __tablename__ = "kpi_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    snapshot_date: Mapped[date]
    metric: Mapped[str]
    dimensions: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    value: Mapped[Decimal] = mapped_column(Numeric(18, 4))

    __table_args__ = (
        Index("ix_kpi_snapshots_date_metric", "snapshot_date", "metric"),
        UniqueConstraint("snapshot_date", "metric", "dimensions", name="uq_kpi_point"),
    )
