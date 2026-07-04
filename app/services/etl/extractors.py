"""Extractors: bring source data in as pandas DataFrames.

Three source kinds per the report (§2.4.1): flat files (CSV/Excel), external
REST APIs, and relational databases.
"""

import io
from typing import Any

import httpx
import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # documented limit (docs/plan/phase-2 risk table)


def extract_tabular(data: bytes, file_name: str) -> pd.DataFrame:
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("file exceeds the 50 MB upload limit")
    name = file_name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    if name.endswith(".csv"):
        try:
            return pd.read_csv(io.BytesIO(data))
        except (pd.errors.ParserError, UnicodeDecodeError) as e:
            raise ValueError(f"could not parse CSV: {e}") from e
    raise ValueError(f"unsupported file type: {file_name} (use .csv, .xlsx, .xls)")


async def extract_rest_api(
    config: dict[str, Any], client: httpx.AsyncClient | None = None
) -> pd.DataFrame:
    """Pull JSON records from `config['url']`; optional dot-path `records_path`."""
    url = config.get("url")
    if not url:
        raise ValueError("rest_api source config needs a 'url'")
    own_client = client is None
    client = client or httpx.AsyncClient(timeout=30)
    try:
        resp = await client.get(url, headers=config.get("headers") or {})
        resp.raise_for_status()
        payload = resp.json()
    finally:
        if own_client:
            await client.aclose()

    for part in (config.get("records_path") or "").split("."):
        if part:
            payload = payload[part]
    if not isinstance(payload, list):
        raise ValueError("REST source did not return a list of records")
    return pd.DataFrame(payload)


async def extract_postgres(config: dict[str, Any]) -> pd.DataFrame:
    """Pull rows from an external Postgres via `config['dsn']` + `config['query']`."""
    dsn, query = config.get("dsn"), config.get("query")
    if not dsn or not query:
        raise ValueError("postgres source config needs 'dsn' and 'query'")
    if not query.lstrip().lower().startswith("select"):
        raise ValueError("postgres source query must be a SELECT")
    engine = create_async_engine(dsn, connect_args={"statement_cache_size": 0})
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text(query))
            rows = result.mappings().all()
    finally:
        await engine.dispose()
    return pd.DataFrame([dict(r) for r in rows])
