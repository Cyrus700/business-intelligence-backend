import io
from datetime import date, timedelta
from typing import Annotated, Any, Literal
from uuid import UUID

import pandas as pd
from fastapi import APIRouter, Depends, Form, HTTPException, Query, Response, UploadFile, status
from sqlalchemy import func, select

from app.api.deps import CurrentUser, DbSession, require_role
from app.models import RawUpload
from app.schemas.integration import PaginatedUploads, TargetDomain, UploadOut
from app.services.etl.extractors import (
    MAX_UPLOAD_BYTES,
    extract_tabular,
    sanitize_upload_filename,
)
from app.services.etl.pipeline import run_frame_pipeline
from app.services.storage import FileStorage, make_key

router = APIRouter(
    prefix="/uploads",
    tags=["data-integration"],
    dependencies=[Depends(require_role("manager"))],
)


def _sample_rows(domain: str) -> list[list[str]]:
    """Header + example rows per domain. Dates are relative to today so the
    sample always passes validation."""
    today = date.today()

    def d(days_back: int) -> str:
        return (today - timedelta(days=days_back)).isoformat()

    return {
        "sales": [
            ["date", "sku", "product_name", "category", "quantity", "unit_price",
             "discount", "customer", "channel", "region"],
            [d(5), "DRY-001", "Basmati Rice 25kg", "Staples", 5, 3600, 0,
             "Bhatbhateni Retail KTM", "wholesale", "Bagmati"],
            [d(4), "BEV-001", "Everest Tea 500g", "Beverages", 12, 320, 20,
             "Namaste Mart", "retail", "Bagmati"],
            [d(3), "SNK-001", "Wai Wai Noodles (30pk)", "Snacks", 8, 640, 0,
             "Daraz Online Nepal", "online", "Bagmati"],
            [d(2), "HHD-001", "Detergent Powder 3kg", "Household", 6, 620, 0,
             "Gurung Kirana Pasal", "retail", "Gandaki"],
            [d(1), "ELC-002", "Electric Kettle 2L", "Electronics", 2, 2350, 50,
             "Everest Traders", "wholesale", "Koshi"],
        ],
        "finance": [
            ["date", "category", "amount", "department", "description"],
            [d(5), "rent", 75000, "Operations", "Warehouse rent"],
            [d(4), "salaries", 215000, "HR", "Monthly payroll"],
            [d(3), "utilities", 18250, "Operations", "Electricity bill"],
            [d(2), "marketing", 45000, "Marketing", "Festival campaign ads"],
            [d(1), "logistics", 23000, "Logistics", "Delivery fleet fuel"],
        ],
        "inventory": [
            ["date", "sku", "product_name", "quantity_on_hand", "reorder_level", "warehouse"],
            [d(2), "DRY-001", "Basmati Rice 25kg", 240, 60, "main"],
            [d(2), "BEV-001", "Everest Tea 500g", 150, 40, "main"],
            [d(2), "SNK-001", "Wai Wai Noodles (30pk)", 90, 50, "main"],
            [d(2), "ELC-001", "Rice Cooker 1.8L", 12, 8, "main"],
            [d(2), "FES-001", "Diyo & Batti Set", 35, 20, "main"],
        ],
    }[domain]


@router.get("/samples/{domain}")
async def sample_template(
    domain: TargetDomain,
    format: Literal["csv", "xlsx"] = Query("csv"),
) -> Response:
    """Download a ready-to-use sample template (header + example rows) for a domain."""
    rows = _sample_rows(domain)
    filename = f"sample_{domain}.{format}"
    if format == "csv":
        content = "\n".join(",".join(str(c) for c in row) for row in rows).encode("utf-8-sig")
        media_type = "text/csv"
    else:
        frame = pd.DataFrame(rows[1:], columns=rows[0])
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            frame.to_excel(writer, index=False, sheet_name=domain.title())
        content = buffer.getvalue()
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _report(
    *,
    upload: RawUpload,
    target_domain: str,
    kind: str | None,
    encoding: str | None,
    columns: list[str] | None,
    preview: list[dict[str, str]] | None,
    warnings: list[str] | None,
    extra: dict[str, Any] | None = None,
) -> dict:
    """Assemble the persisted validation report for an upload."""
    report: dict = {"target_domain": target_domain}
    if kind is not None:
        report["kind"] = kind
    if encoding is not None:
        report["encoding"] = encoding
    if columns is not None:
        report["columns"] = columns
    if preview is not None:
        report["preview"] = preview
    if warnings:
        report["warnings"] = warnings
    # Use __dict__ to avoid lazy-load MissingGreenlet after commit expiry
    _err = upload.__dict__.get("error_report")
    if _err and "error" in _err:
        report["error"] = _err["error"]
    if extra:
        report.update(extra)
    return report


@router.post("", response_model=UploadOut, status_code=status.HTTP_201_CREATED)
async def upload_file(
    db: DbSession,
    user: CurrentUser,
    file: UploadFile,
    domain: Annotated[TargetDomain, Form()],
    data_source_id: Annotated[UUID | None, Form()] = None,
) -> UploadOut:
    """Upload a CSV/Excel file; it is validated, transformed, and loaded synchronously,
    and the full validation report is returned."""
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "file exceeds 50 MB")
    file_name = sanitize_upload_filename(file.filename or "upload.csv")

    from app.api.deps import user_org_id

    upload = RawUpload(
        data_source_id=data_source_id,
        file_name=file_name,
        uploaded_by=user.id,
        status="received",
        target_domain=domain,
        org_id=user_org_id(user),
    )
    db.add(upload)
    await db.flush()

    upload.s3_key = FileStorage().save(make_key(file_name), data)

    try:
        extract = extract_tabular(data, file_name)
    except ValueError as e:
        upload.status = "failed"
        upload.error_report = {"error": str(e)}
        await db.commit()
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e)) from e

    upload.status = "validated"
    upload.row_count = len(extract.frame)

    try:
        result = await run_frame_pipeline(
            db, domain, extract.frame, trigger="upload", source_id=data_source_id, org_id=user_org_id(user)
        )
    except ValueError as e:  # e.g. missing required columns
        upload.status = "failed"
        upload.error_report = _report(
            upload=upload,
            target_domain=domain,
            kind=extract.kind,
            encoding=extract.encoding,
            columns=extract.columns,
            preview=extract.preview,
            warnings=extract.warnings,
            extra={"error": str(e)},
        )
        await db.commit()
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e)) from e

    upload.status = "loaded"
    upload.error_report = _report(
        upload=upload,
        target_domain=domain,
        kind=extract.kind,
        encoding=extract.encoding,
        columns=extract.columns,
        preview=extract.preview,
        warnings=extract.warnings,
        extra={
            "loaded": result.rows_loaded,
            "rejected": result.rows_rejected,
            "skipped_duplicates": result.skipped_duplicates,
            "details": result.error_report.get("details", []),
            "file_size": len(data),
        },
    )
    await db.commit()
    await db.refresh(upload)
    out = UploadOut.model_validate(upload)
    out.etl_job_id = result.job_id
    return out


@router.get("", response_model=PaginatedUploads)
async def list_uploads(
    db: DbSession,
    user: CurrentUser,
    status_filter: str | None = Query(None, alias="status"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> PaginatedUploads:
    from app.api.deps import org_predicate

    stmt = select(RawUpload).where(org_predicate(RawUpload.org_id, user.org_id))
    if status_filter:
        stmt = stmt.where(RawUpload.status == status_filter)
    total = (
        (await db.execute(select(func.count()).select_from(stmt.subquery()))).scalar_one() or 0
    )
    rows = (
        (await db.execute(stmt.offset((page - 1) * page_size).limit(page_size))).scalars().all()
    )
    return PaginatedUploads(
        items=[UploadOut.model_validate(r) for r in rows],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{upload_id}", response_model=UploadOut)
async def get_upload(
    upload_id: UUID,
    db: DbSession,
    user: CurrentUser,
) -> UploadOut:
    from app.api.deps import org_predicate

    upload = (
        await db.execute(
            select(RawUpload).where(
                RawUpload.id == upload_id, org_predicate(RawUpload.org_id, user.org_id)
            )
        )
    ).scalar_one_or_none()
    if upload is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Upload not found")
    return UploadOut.model_validate(upload)
