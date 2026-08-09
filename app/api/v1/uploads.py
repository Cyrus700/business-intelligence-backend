from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import func, select

from app.api.deps import CurrentUser, DbSession, require_role
from app.models import RawUpload
from app.schemas.integration import PaginatedUploads, TargetDomain, UploadOut
from app.services.etl.extractors import MAX_UPLOAD_BYTES, extract_tabular
from app.services.etl.pipeline import run_frame_pipeline
from app.services.storage import FileStorage, make_key

router = APIRouter(
    prefix="/uploads",
    tags=["data-integration"],
    dependencies=[Depends(require_role("manager"))],
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
    if upload.error_report and "error" in upload.error_report:
        report["error"] = upload.error_report["error"]
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
    file_name = file.filename or "upload.csv"

    upload = RawUpload(
        data_source_id=data_source_id,
        file_name=file_name,
        uploaded_by=user.id,
        status="received",
        target_domain=domain,
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
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e)) from e

    upload.status = "validated"
    upload.row_count = len(extract.frame)

    try:
        result = await run_frame_pipeline(
            db, domain, extract.frame, trigger="upload", source_id=data_source_id
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
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e)) from e

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
    status_filter: str | None = Query(None, alias="status"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> PaginatedUploads:
    stmt = select(RawUpload).order_by(RawUpload.created_at.desc())
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
    upload_id: UUID, db: DbSession, _: Annotated[object, Depends(require_role("analyst"))]
) -> UploadOut:
    upload = await db.get(RawUpload, upload_id)
    if upload is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Upload not found")
    return UploadOut.model_validate(upload)
