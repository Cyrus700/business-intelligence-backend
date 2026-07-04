import httpx
import pytest

from app.services.etl.extractors import extract_rest_api, extract_tabular

CSV = b"date,sku,quantity,unit_price\n2026-06-15,BEV-001,3,320\n"


def test_extract_csv():
    frame = extract_tabular(CSV, "sales.csv")
    assert list(frame.columns) == ["date", "sku", "quantity", "unit_price"]
    assert len(frame) == 1


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
