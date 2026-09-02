from decimal import Decimal

import pandas as pd
import pytest

from app.services.etl.domains import (
    transform_expense_row,
    transform_frame,
    transform_sales_row,
)


def _sales_row(**overrides):
    row = {
        "date": "2026-06-15",
        "sku": "BEV-001",
        "product_name": "Everest Tea 500g",
        "category": "Beverages",
        "customer": "Namaste Mart",
        "segment": "retail",
        "city": "Kathmandu",
        "region": "Bagmati",
        "channel": "store",
        "quantity": 3,
        "unit_price": 320,
        "discount": 0,
    }
    row.update(overrides)
    return row


def test_sales_row_happy_path():
    record = transform_sales_row(_sales_row())
    assert record["total_amount"] == Decimal("960.00")
    assert record["row_hash"]


def test_sales_row_hash_is_deterministic_and_field_sensitive():
    a = transform_sales_row(_sales_row())
    b = transform_sales_row(_sales_row())
    c = transform_sales_row(_sales_row(quantity=4))
    assert a["row_hash"] == b["row_hash"]
    assert a["row_hash"] != c["row_hash"]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"date": "not-a-date"}, "unparseable date"),
        ({"date": "2099-01-01"}, "out of range"),
        ({"sku": ""}, "sku is missing"),
        ({"quantity": 0}, "quantity"),
        ({"quantity": -2}, "quantity"),
        ({"unit_price": "free"}, "unit_price"),
        ({"discount": 99999}, "exceeds"),
    ],
)
def test_sales_row_rejections(overrides, match):
    with pytest.raises(ValueError, match=match):
        transform_sales_row(_sales_row(**overrides))


def test_expense_row_category_enum():
    good = transform_expense_row({"date": "2026-06-01", "category": "Rent", "amount": 185000})
    assert good["category"] == "rent"
    with pytest.raises(ValueError, match="unknown expense category"):
        transform_expense_row({"date": "2026-06-01", "category": "bribes", "amount": 5})
    with pytest.raises(ValueError, match="amount"):
        transform_expense_row({"date": "2026-06-01", "category": "rent", "amount": 0})


def test_transform_frame_missing_columns():
    with pytest.raises(ValueError, match="missing required columns: quantity, unit_price"):
        transform_frame("sales", pd.DataFrame([{"date": "2026-01-01", "sku": "X"}]))


def test_transform_frame_collects_row_errors():
    frame = pd.DataFrame([_sales_row(), _sales_row(quantity=-1), _sales_row(date="bad")])
    result = transform_frame("sales", frame)
    assert len(result.records) == 1
    assert [e.row for e in result.errors] == [2, 3]
    assert result.error_report["rejected"] == 2


def test_transform_frame_normalises_column_case():
    frame = pd.DataFrame([{k.upper(): v for k, v in _sales_row().items()}])
    result = transform_frame("sales", frame)
    assert len(result.records) == 1


def test_transform_frame_accepts_alias_columns():
    frame = pd.DataFrame(
        [
            {"Qty": 2, "Item Code": "BEV-001", "Price": 320, "Txn Date": "2026-06-15"},
            {"Qty": 5, "Item Code": "BEV-002", "Price": 120, "Txn Date": "2026-06-16"},
        ]
    )
    result = transform_frame("sales", frame)
    assert len(result.records) == 2
    assert result.records[0]["sku"] == "BEV-001"
    assert result.records[0]["quantity"] == 2
    assert result.records[0]["unit_price"] == Decimal("320.00")


def test_transform_frame_alias_for_expenses_and_inventory():
    expenses = transform_frame(
        "finance", pd.DataFrame([{"Expense Date": "2026-06-01", "Expense Category": "Rent", "Value": 5000}])
    )
    assert len(expenses.records) == 1
    assert expenses.records[0]["amount"] == Decimal("5000.00")
    inventory = transform_frame("inventory", pd.DataFrame([{"Date": "2026-06-01", "Item Code": "A-1", "Stock": 42}]))
    assert len(inventory.records) == 1
    assert inventory.records[0]["quantity_on_hand"] == 42


def test_transform_frame_rejects_conflicting_alias_columns():
    frame = pd.DataFrame([{"qty": 1, "quantity": 2, "date": "2026-01-01", "sku": "X", "unit_price": 5}])
    with pytest.raises(ValueError, match="duplicate column names"):
        transform_frame("sales", frame)
