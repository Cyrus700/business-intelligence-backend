from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, status

from app.api.deps import CurrentUser, DbSession, require_role
from app.models import RawUpload
from app.schemas.integration import TargetDomain, UploadOut
from app.services.etl.extractors import MAX_UPLOAD_BYTES, extract_tabular
from app.services.etl.pipeline import run_frame_pipeline
from app.services.storage import FileStorage, make_key

router = APIRouter(
    prefix="/uploads",
    tags=["data-integration"],
    dependencies=[Depends(require_role("manager"))],
)


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
    )
    db.add(upload)
    await db.flush()

    upload.s3_key = FileStorage().save(make_key(file_name), data)

    try:
        frame = extract_tabular(data, file_name)
    except ValueError as e:
        upload.status = "failed"
        upload.error_report = {"error": str(e)}
        await db.commit()
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e)) from e

    upload.row_count = len(frame)
    try:
        result = await run_frame_pipeline(
            db, domain, frame, trigger="upload", source_id=data_source_id
        )
    except ValueError as e:  # e.g. missing required columns
        upload.status = "failed"
        upload.error_report = {"error": str(e)}
        await db.commit()
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e)) from e

    upload.status = "loaded"
    upload.error_report = {
        "loaded": result.rows_loaded,
        "skipped_duplicates": result.skipped_duplicates,
        **result.error_report,
    }
    await db.commit()
    await db.refresh(upload)
    out = UploadOut.model_validate(upload)
    out.etl_job_id = result.job_id
    return out


@router.get("/{upload_id}", response_model=UploadOut)
async def get_upload(
    upload_id: UUID, db: DbSession, _: Annotated[object, Depends(require_role("analyst"))]
) -> UploadOut:
    upload = await db.get(RawUpload, upload_id)
    if upload is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Upload not found")
    return UploadOut.model_validate(upload)
