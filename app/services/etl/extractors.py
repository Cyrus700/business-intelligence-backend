"""Extractors: bring source data in as pandas DataFrames.

Three source kinds per the report (§2.4.1): flat files (CSV/Excel), external
REST APIs, and relational databases.
"""

import csv
import io
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import httpx
import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # documented limit (docs/plan/phase-2 risk table)

_PREVIEW_ROWS = 5


@dataclass
class TabularExtract:
    """Validated tabular file: the DataFrame plus metadata for the validation report."""

    frame: pd.DataFrame
    file_name: str
    kind: str  # "csv" | "excel"
    encoding: str | None  # detected encoding for CSV, None for Excel
    columns: list[str]
    preview: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Strip + lowercase column names (mirrors transform_frame) and reject duplicates."""
    normalized = [str(c).strip().lower() for c in frame.columns]
    duplicates = sorted({c for c in normalized if normalized.count(c) > 1})
    if duplicates:
        raise ValueError(f"duplicate column names: {', '.join(duplicates)}")
    frame.columns = normalized
    return frame


def _preview_rows(frame: pd.DataFrame, n: int = _PREVIEW_ROWS) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for _, record in frame.head(n).iterrows():
        row: dict[str, str] = {}
        for col, value in record.items():
            if pd.isna(value):
                row[str(col)] = ""
            elif isinstance(value, (pd.Timestamp, datetime, date)):
                row[str(col)] = value.isoformat()
            else:
                row[str(col)] = str(value)[:120]
        rows.append(row)
    return rows


def _detect_encoding(data: bytes) -> str:
    """Sniff CSV encoding: UTF-8 BOM, strict UTF-8, then cp1252 as a fallback."""
    if data.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        data.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "cp1252"


def _detect_duplicate_header(data: bytes, encoding: str) -> None:
    """Reject duplicate CSV column names before pandas can mangle them (date, date.1)."""
    first_line = data.decode(encoding).splitlines()
    if not first_line:
        return
    columns = [c.strip().lower() for c in next(csv.reader([first_line[0]]))]
    duplicates = sorted({c for c in columns if columns.count(c) > 1})
    if duplicates:
        raise ValueError(f"duplicate column names: {', '.join(duplicates)}")


def sanitize_upload_filename(file_name: str) -> str:
    """Keep only a safe basename with a whitelisted extension."""
    base = os.path.basename(file_name.replace("\\", "/")).strip()
    if not base or base in (".", "..") or base.startswith("."):
        raise ValueError("invalid file name")
    lower = base.lower()
    if not lower.endswith((".csv", ".xlsx", ".xls")):
        raise ValueError("unsupported file type (use .csv, .xlsx or .xls)")
    return base


def extract_tabular(data: bytes, file_name: str) -> TabularExtract:
    """Parse + validate an uploaded CSV/Excel file.

    Returns the frame with normalized column names plus a metadata report
    (columns, preview rows, warnings, encoding). Raises ValueError with a
    human-readable reason for anything that cannot be loaded.
    """
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("file exceeds the 50 MB upload limit")
    if not data or not data.strip():
        raise ValueError("file is empty (0 bytes)")
    name = file_name.lower()
    warnings: list[str] = []

    if name.endswith(".xlsx"):
        if not data.startswith(b"PK\x03\x04"):
            raise ValueError(
                "file looks like an .xlsx but its content is not a ZIP/OOXML workbook"
            )
    elif name.endswith(".xls"):
        if not data.startswith(b"\xd0\xcf\x11\xe0"):
            raise ValueError("file looks like an .xls but its content is not an OLE2 workbook")
    elif name.endswith(".csv"):
        # A CSV must not be a ZIP (xlsx misnamed), an OLE2 document or a web
        # page — anything else is a disguised file we refuse to parse.
        if data.startswith((b"PK\x03\x04", b"\xd0\xcf\x11\xe0")) or b"<html" in data[:4096].lower():
            raise ValueError("file looks like a .csv but its content is not a CSV")

    if name.endswith((".xlsx", ".xls")):
        try:
            frame = pd.read_excel(io.BytesIO(data))
        except Exception as e:  # noqa: BLE001 - surface any workbook error to the user
            raise ValueError(f"could not read Excel workbook: {str(e)[:200]}") from e
        if frame.empty:
            raise ValueError("Excel workbook contains no data rows")
        frame = _normalize_columns(frame)
        if frame.empty:
            raise ValueError("Excel workbook contains a header but no data rows")
        return TabularExtract(
            frame=frame,
            file_name=file_name,
            kind="excel",
            encoding=None,
            columns=list(frame.columns),
            preview=_preview_rows(frame),
            warnings=warnings,
        )

    if name.endswith(".csv"):
        encoding = _detect_encoding(data)
        if encoding == "cp1252":
            warnings.append(
                "file is not UTF-8 encoded (looks like Windows-1252/Latin-1); decoded automatically"
            )
        _detect_duplicate_header(data, encoding)
        try:
            frame = pd.read_csv(io.BytesIO(data), encoding=encoding)
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as e:
            raise ValueError(f"could not parse CSV: {str(e)[:200]}") from e
        if frame.empty:
            raise ValueError("CSV contains a header but no data rows")
        frame = _normalize_columns(frame)
        return TabularExtract(
            frame=frame,
            file_name=file_name,
            kind="csv",
            encoding=encoding,
            columns=list(frame.columns),
            preview=_preview_rows(frame),
            warnings=warnings,
        )

    raise ValueError(f"unsupported file type: {file_name} (use .csv, .xlsx, .xls)")


async def extract_rest_api(
    config: dict[str, Any], client: httpx.AsyncClient | None = None
) -> pd.DataFrame:
    """Pull JSON records from `config['url']`; optional dot-path `records_path`."""
    url = config.get("url")
    if not url:
        raise ValueError("rest_api source config needs a 'url'")
    from app.services.etl.ssrf import validate_public_http_url

    validate_public_http_url(url)
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
