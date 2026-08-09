import io

import httpx
import pandas as pd
import pytest

from app.services.etl.extractors import extract_rest_api, extract_tabular

CSV = b"date,sku,quantity,unit_price\n2026-06-15,BEV-001,3,320\n"


def test_extract_csv():
    extract = extract_tabular(CSV, "sales.csv")
    assert extract.kind == "csv"
    assert extract.encoding == "utf-8"
    assert extract.columns == ["date", "sku", "quantity", "unit_price"]
    assert len(extract.frame) == 1
    assert extract.preview[0]["sku"] == "BEV-001"
    assert extract.warnings == []


def test_extract_csv_normalizes_columns():
    extract = extract_tabular(b"Date,Sku,Quantity\n2026-06-15,A,1\n", "sales.csv")
    assert extract.columns == ["date", "sku", "quantity"]


def test_extract_csv_with_utf8_bom():
    extract = extract_tabular(b"\xef\xbb\xbfdate,sku\n2026-06-15,A\n", "sales.csv")
    assert extract.encoding == "utf-8-sig"
    assert extract.columns == ["date", "sku"]


def test_extract_csv_cp1252_encoding_with_warning():
    raw = "date,sku,product_name\n2026-06-15,A,Café\n".encode("cp1252")
    extract = extract_tabular(raw, "sales.csv")
    assert extract.encoding == "cp1252"
    assert extract.warnings  # decode-fallback warning present
    assert extract.frame.iloc[0]["product_name"] == "Café"


def test_extract_csv_rejects_duplicate_columns():
    with pytest.raises(ValueError, match="duplicate column names"):
        extract_tabular(b"date,sku,date\n2026-06-15,A,1\n", "sales.csv")


def test_extract_csv_rejects_empty_file():
    with pytest.raises(ValueError, match="file is empty"):
        extract_tabular(b"", "sales.csv")


def test_extract_csv_rejects_header_only():
    with pytest.raises(ValueError, match="no data rows"):
        extract_tabular(b"date,sku,quantity,unit_price\n", "sales.csv")


def test_extract_excel():
    frame = pd.DataFrame(
        {"date": ["2026-06-15"], "sku": ["BEV-001"], "quantity": [3], "unit_price": [320]}
    )
    buf = io.BytesIO()
    frame.to_excel(buf, index=False)
    extract = extract_tabular(buf.getvalue(), "sales.xlsx")
    assert extract.kind == "excel"
    assert extract.encoding is None
    assert list(extract.frame["sku"]) == ["BEV-001"]
    assert extract.preview[0]["sku"] == "BEV-001"


def test_extract_excel_rejects_empty_workbook():
    buf = io.BytesIO()
    pd.DataFrame().to_excel(buf, index=False)
    with pytest.raises(ValueError, match="no data rows"):
        extract_tabular(buf.getvalue(), "sales.xlsx")


def test_extract_rejects_unknown_extension():
    with pytest.raises(ValueError, match="unsupported file type"):
        extract_tabular(CSV, "sales.pdf")


def test_extract_rejects_oversize():
    with pytest.raises(ValueError, match="50 MB"):
        extract_tabular(b"x" * (51 * 1024 * 1024), "big.csv")


async def test_extract_rest_api_with_records_path():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"data": {"rows": [{"date": "2026-06-15", "sku": "A", "quantity": 1}]}}
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    frame = await extract_rest_api(
        {"url": "https://pos.example.com/export", "records_path": "data.rows"}, client
    )
    await client.aclose()
    assert len(frame) == 1
    assert frame.iloc[0]["sku"] == "A"


async def test_extract_rest_api_rejects_non_list():
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"ok": True}))
    )
    with pytest.raises(ValueError, match="list of records"):
        await extract_rest_api({"url": "https://x.example.com"}, client)
    await client.aclose()
