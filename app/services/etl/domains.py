"""Per-domain validation + transformation: raw DataFrame rows → canonical records.

Each domain spec declares required columns and a row transformer that either
returns a canonical dict (with a deterministic row_hash for idempotent loading)
or raises ValueError with a human-readable reason.
"""

import hashlib
from collections.abc import Callable
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

import pandas as pd

from app.services.etl.base import RowError, TransformResult

EXPENSE_CATEGORIES = {"rent", "salaries", "utilities", "marketing", "logistics", "other"}
MIN_DATE = date(2000, 1, 1)


def _parse_date(value: Any) -> date:
    if pd.isna(value):
        raise ValueError("date is missing")
    if isinstance(value, datetime):
        parsed = value.date()
    elif isinstance(value, date):
        parsed = value
    else:
        try:
            parsed = pd.to_datetime(str(value), format="mixed", dayfirst=False).date()
        except (ValueError, TypeError) as e:
            raise ValueError(f"unparseable date: {value!r}") from e
    if parsed < MIN_DATE or parsed > date.today():
        raise ValueError(f"date out of range: {parsed.isoformat()}")
    return parsed


def _parse_money(value: Any, field: str, *, allow_zero: bool = True) -> Decimal:
    try:
        amount = Decimal(str(value)).quantize(Decimal("0.01"))
    except (InvalidOperation, TypeError, ValueError) as e:
        raise ValueError(f"{field} is not a number: {value!r}") from e
    if amount < 0 or (amount == 0 and not allow_zero):
        raise ValueError(f"{field} must be {'>=' if allow_zero else '>'} 0, got {amount}")
    return amount


def _parse_int(value: Any, field: str, *, minimum: int = 0) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError) as e:
        raise ValueError(f"{field} is not an integer: {value!r}") from e
    if number < minimum:
        raise ValueError(f"{field} must be >= {minimum}, got {number}")
    return number


def _text(value: Any) -> str | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    return str(value).strip()


def _row_hash(*parts: Any) -> str:
    canonical = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(canonical.encode()).hexdigest()


def transform_sales_row(raw: dict[str, Any]) -> dict[str, Any]:
    txn_date = _parse_date(raw.get("date"))
    sku = _text(raw.get("sku"))
    if not sku:
        raise ValueError("sku is missing")
    quantity = _parse_int(raw.get("quantity"), "quantity", minimum=1)
    unit_price = _parse_money(raw.get("unit_price"), "unit_price")
    discount = _parse_money(
        raw.get("discount", 0) if not pd.isna(raw.get("discount", 0)) else 0, "discount"
    )
    total = (unit_price * quantity - discount).quantize(Decimal("0.01"))
    if total < 0:
        raise ValueError(f"discount {discount} exceeds line total")
    customer = _text(raw.get("customer"))
    record = {
        "txn_date": txn_date,
        "sku": sku,
        "product_name": _text(raw.get("product_name")) or sku,
        "category": _text(raw.get("category")),
        "customer_name": customer,
        "segment": _text(raw.get("segment")),
        "city": _text(raw.get("city")),
        "region": _text(raw.get("region")),
        "channel": _text(raw.get("channel")),
        "quantity": quantity,
        "unit_price": unit_price,
        "discount": discount,
        "total_amount": total,
    }
    record["row_hash"] = _row_hash(
        "sales",
        txn_date.isoformat(),
        sku,
        customer,
        record["channel"],
        record["region"],
        quantity,
        unit_price,
        discount,
    )
    return record


def transform_expense_row(raw: dict[str, Any]) -> dict[str, Any]:
    expense_date = _parse_date(raw.get("date"))
    category = (_text(raw.get("category")) or "").lower()
    if category not in EXPENSE_CATEGORIES:
        raise ValueError(f"unknown expense category: {raw.get('category')!r}")
    amount = _parse_money(raw.get("amount"), "amount", allow_zero=False)
    record = {
        "expense_date": expense_date,
        "category": category,
        "amount": amount,
        "department": _text(raw.get("department")),
        "description": _text(raw.get("description")),
    }
    record["row_hash"] = _row_hash(
        "expense",
        expense_date.isoformat(),
        category,
        amount,
        record["department"],
        record["description"],
    )
    return record


def transform_inventory_row(raw: dict[str, Any]) -> dict[str, Any]:
    snapshot_date = _parse_date(raw.get("date"))
    sku = _text(raw.get("sku"))
    if not sku:
        raise ValueError("sku is missing")
    return {
        "snapshot_date": snapshot_date,
        "sku": sku,
        "quantity_on_hand": _parse_int(raw.get("quantity_on_hand"), "quantity_on_hand"),
        "reorder_level": _parse_int(raw.get("reorder_level", 0), "reorder_level"),
        "warehouse": _text(raw.get("warehouse")) or "main",
    }


DOMAIN_SPECS: dict[str, dict[str, Any]] = {
    "sales": {
        "required_columns": {"date", "sku", "quantity", "unit_price"},
        "transform": transform_sales_row,
    },
    "finance": {
        "required_columns": {"date", "category", "amount"},
        "transform": transform_expense_row,
    },
    "inventory": {
        "required_columns": {"date", "sku", "quantity_on_hand"},
        "transform": transform_inventory_row,
    },
}


def transform_frame(domain: str, frame: pd.DataFrame) -> TransformResult:
    spec = DOMAIN_SPECS[domain]
    frame = frame.rename(columns={c: str(c).strip().lower() for c in frame.columns})
    missing = spec["required_columns"] - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {', '.join(sorted(missing))}")

    transform: Callable[[dict[str, Any]], dict[str, Any]] = spec["transform"]
    result = TransformResult()
    for i, raw in enumerate(frame.to_dict("records"), start=1):
        try:
            result.records.append(transform(raw))
        except ValueError as e:
            result.errors.append(RowError(row=i, reason=str(e)))
    return result
