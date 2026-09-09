import io
import logging
from datetime import date, timedelta
from typing import Annotated, Any, Literal
from uuid import UUID

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, Query, Response, UploadFile, status
from sqlalchemy import func, select

from app.api.deps import CurrentUser, DbSession, require_role
from app.core.database import get_session_factory
from app.models import RawUpload
from app.schemas.integration import PaginatedUploads, TargetDomain, UploadOut
from app.services.etl.agents import DOMAIN_BUSINESS_META, orchestrator
from app.services.etl.extractors import (
    MAX_UPLOAD_BYTES,
    extract_tabular,
    sanitize_upload_filename,
)
from app.services.etl.pipeline import run_frame_pipeline
from app.services.storage import FileStorage, make_key

logger = logging.getLogger(__name__)

# Threshold where we switch to async background processing to avoid gateway timeouts
ASYNC_THRESHOLD_BYTES = 5 * 1024 * 1024  # 5 MB
ASYNC_THRESHOLD_ROWS = 10000

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
            [
                "date",
                "sku",
                "product_name",
                "category",
                "quantity",
                "unit_price",
                "discount",
                "customer",
                "channel",
                "region",
            ],
            [
                d(5),
                "DRY-001",
                "Basmati Rice 25kg",
                "Staples",
                5,
                3600,
                0,
                "Bhatbhateni Retail KTM",
                "wholesale",
                "Bagmati",
            ],
            [d(4), "BEV-001", "Everest Tea 500g", "Beverages", 12, 320, 20, "Namaste Mart", "retail", "Bagmati"],
            [d(3), "SNK-001", "Wai Wai Noodles (30pk)", "Snacks", 8, 640, 0, "Daraz Online Nepal", "online", "Bagmati"],
            [
                d(2),
                "HHD-001",
                "Detergent Powder 3kg",
                "Household",
                6,
                620,
                0,
                "Gurung Kirana Pasal",
                "retail",
                "Gandaki",
            ],
            [
                d(1),
                "ELC-002",
                "Electric Kettle 2L",
                "Electronics",
                2,
                2350,
                50,
                "Everest Traders",
                "wholesale",
                "Koshi",
            ],
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


async def _process_upload_background(
    upload_id: UUID,
    domain: str,
    data: bytes,
    file_name: str,
    org_id: UUID | None,
    data_source_id: UUID | None = None,
) -> None:
    """Background worker for large files — runs outside the request lifecycle."""
    factory = get_session_factory()
    async with factory() as db:
        upload = await db.get(RawUpload, upload_id)
        if not upload:
            logger.error("background upload %s not found", upload_id)
            return
        try:
            extract = extract_tabular(data, file_name)
            upload.row_count = len(extract.frame)
            upload.error_report = _report(
                upload=upload,
                target_domain=domain,
                kind=extract.kind,
                encoding=extract.encoding,
                columns=extract.columns,
                preview=extract.preview,
                warnings=extract.warnings,
            )
            await db.flush()
            result = await run_frame_pipeline(
                db, domain, extract.frame, trigger="upload", source_id=data_source_id, org_id=org_id
            )
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
            # attach job id
            upload.error_report["etl_job_id"] = result.job_id
            await db.commit()
            logger.info("background upload %s completed: %s rows", upload_id, result.rows_loaded)
        except ValueError as e:
            upload.status = "failed"
            # preserve prior report if exists
            prior = upload.error_report or {}
            prior["error"] = str(e)
            upload.error_report = prior
            await db.commit()
            logger.warning("background upload %s failed: %s", upload_id, e)
        except Exception as e:
            logger.exception("background upload %s crashed", upload_id)
            upload.status = "failed"
            upload.error_report = {"error": f"internal error: {str(e)[:300]}"}
            await db.commit()


@router.post("/inspect", status_code=status.HTTP_200_OK)
async def inspect_file(
    user: CurrentUser,
    file: UploadFile,
) -> dict[str, Any]:
    """Inspect a file *without* loading it — returns type, columns, and domain suggestion.

    This is the FileInspectorAgent endpoint. The browser calls it on file pick to
    auto-detect domain, show column mapping, and surface warnings before upload.
    No DB write, no authz beyond manager role (inherited from router).
    """
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "file exceeds 50 MB")
    if not data or not data.strip():
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "file is empty")
    file_name = sanitize_upload_filename(file.filename or "upload.csv")
    try:
        inspection = orchestrator.inspector.inspect(data, file_name)
    except ValueError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e)) from e

    # Also run a light validation for each domain to help the UI show readiness
    validation_hints: dict[str, Any] = {}
    for d in ("sales", "finance", "inventory"):
        try:
            from app.services.etl.domains import DOMAIN_SPECS

            # quick check: does it have required columns?
            required = DOMAIN_SPECS[d]["required_columns"]
            canonical = set(inspection.canonical_columns)
            missing = sorted(required - canonical)
            validation_hints[d] = {
                "ready": len(missing) == 0,
                "missing": missing,
                "confidence": inspection.detected["scores"][d]["confidence"],
            }
        except Exception:
            validation_hints[d] = {"ready": False, "missing": [], "confidence": 0}

    # Business ease: let DomainIntelligence explain in plain English
    intel = orchestrator.domain_intel.explain(inspection.detected)
    return {
        "file_name": file_name,
        "kind": inspection.kind,
        "mime_hint": inspection.mime_hint,
        "encoding": inspection.encoding,
        "size_bytes": inspection.size_bytes,
        "columns": inspection.columns,
        "canonical_columns": inspection.canonical_columns,
        "preview": inspection.preview,
        "warnings": inspection.warnings,
        "detected": inspection.detected,
        "validation": validation_hints,
        "row_estimate": inspection.row_estimate,
        "sheet_name": inspection.sheet_name,
        "business_summary": inspection.business_summary,
        "business_meta": inspection.business_meta,
        "quality_hints": inspection.quality_hints,
        "intel": intel,
        "all_domain_meta": DOMAIN_BUSINESS_META,
    }


# ---- Chunked upload: small→large professional path ----------------------

@router.post("/chunked/init", status_code=status.HTTP_201_CREATED)
async def chunked_init(
    user: CurrentUser,
    file_name: str = Form(...),
    total_size: int = Form(...),
    total_chunks: int = Form(...),
    domain: str | None = Form(None),
) -> dict[str, Any]:
    """Start a chunked upload session. Returns session_id for subsequent chunk posts."""
    if total_size > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "file exceeds 50 MB")
    safe_name = sanitize_upload_filename(file_name)
    meta = orchestrator.chunk_manager.init_session(safe_name, domain, total_size, total_chunks)
    return {"session_id": meta["session_id"], "chunk_size": 1 * 1024 * 1024}


@router.post("/chunked/{session_id}/chunk", status_code=status.HTTP_200_OK)
async def chunked_chunk(
    session_id: str,
    user: CurrentUser,
    chunk_index: Annotated[int, Form()],
    chunk: Annotated[UploadFile, Form()],
) -> dict[str, Any]:
    """Upload a single chunk (1 MB recommended)."""
    data = await chunk.read()
    try:
        received = orchestrator.chunk_manager.store_chunk(session_id, chunk_index, data)
    except ValueError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e)) from e
    return {"received": received, "chunk_index": chunk_index}


@router.post("/chunked/{session_id}/complete", response_model=UploadOut, status_code=status.HTTP_201_CREATED)
async def chunked_complete(
    session_id: str,
    db: DbSession,
    user: CurrentUser,
    background_tasks: BackgroundTasks,
    domain: Annotated[TargetDomain | None, Form()] = None,
    data_source_id: Annotated[UUID | None, Form()] = None,
) -> UploadOut:
    """Assemble chunks and process. Uses background task for large assemblies."""
    try:
        data, meta = orchestrator.chunk_manager.assemble(session_id)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from e

    file_name = meta["file_name"]
    # Auto-detect domain if not supplied or if supplied domain mismatches strongly
    effective_domain = domain or meta.get("domain")
    if not effective_domain:
        detection = orchestrator.suggested_domain(data, file_name)
        effective_domain = detection.get("suggested")
        if not effective_domain:
            orchestrator.chunk_manager.cleanup(session_id)
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "could not auto-detect domain from columns — please select sales, finance, or inventory",
            )
    # Validate requested domain is allowed
    if effective_domain not in ("sales", "finance", "inventory"):
        orchestrator.chunk_manager.cleanup(session_id)
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"invalid domain: {effective_domain}")

    from app.api.deps import user_org_id

    upload = RawUpload(
        file_name=file_name,
        uploaded_by=user.id,
        status="received",
        target_domain=effective_domain,
        org_id=user_org_id(user),
        data_source_id=data_source_id,
    )
    db.add(upload)
    await db.flush()
    upload.s3_key = FileStorage().save(make_key(file_name), data)
    await db.commit()
    await db.refresh(upload)

    # Decide sync vs async based on size
    use_background = len(data) > ASYNC_THRESHOLD_BYTES

    if use_background:
        # Mark as processing and offload
        upload.status = "received"
        upload.error_report = {"status": "processing", "message": "large file — processing in background", "file_size": len(data)}
        await db.commit()
        background_tasks.add_task(
            _process_upload_background,
            upload.id,
            effective_domain,
            data,
            file_name,
            user_org_id(user),
            data_source_id,
        )
        orchestrator.chunk_manager.cleanup(session_id)
        out = UploadOut.model_validate(upload)
        return out

    # Fast path: process synchronously
    try:
        extract = extract_tabular(data, file_name)
        upload.status = "validated"
        upload.row_count = len(extract.frame)
        await db.flush()
        result = await run_frame_pipeline(
            db, effective_domain, extract.frame, trigger="upload", source_id=data_source_id, org_id=user_org_id(user)
        )
        upload.status = "loaded"
        upload.error_report = _report(
            upload=upload,
            target_domain=effective_domain,
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
        orchestrator.chunk_manager.cleanup(session_id)
        return out
    except ValueError as e:
        upload.status = "failed"
        upload.error_report = {"error": str(e)}
        await db.commit()
        orchestrator.chunk_manager.cleanup(session_id)
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e)) from e


@router.post("", response_model=UploadOut, status_code=status.HTTP_201_CREATED)
async def upload_file(
    db: DbSession,
    user: CurrentUser,
    background_tasks: BackgroundTasks,
    file: UploadFile,
    domain: Annotated[TargetDomain | None, Form()] = None,
    data_source_id: Annotated[UUID | None, Form()] = None,
) -> UploadOut:
    """Upload a CSV/Excel file — now with smart domain detection and large-file support.

    - If `domain` is omitted, the FileInspectorAgent auto-detects it from headers.
    - Small files (≤5 MB, ≤10k rows) are processed synchronously and return 201 with report.
    - Large files are accepted immediately and processed in background; poll GET /uploads/{id}.
    """
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "file exceeds 50 MB")
    if not data or not data.strip():
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "file is empty")
    file_name = sanitize_upload_filename(file.filename or "upload.csv")

    # ---- Agent 1: FileInspector — auto-detect domain if not supplied ----
    effective_domain = domain
    detection = None
    if not effective_domain:
        detection = orchestrator.suggested_domain(data, file_name)
        effective_domain = detection.get("suggested")
        if not effective_domain:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "could not auto-detect domain — file must contain columns for sales, finance, or inventory. Please select a domain.",
            )
    if effective_domain not in ("sales", "finance", "inventory"):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"invalid domain: {effective_domain}")

    from app.api.deps import user_org_id

    # Small vs large: large files get instant 201 + background processing to avoid gateway timeouts
    is_large = len(data) > ASYNC_THRESHOLD_BYTES

    upload = RawUpload(
        data_source_id=data_source_id,
        file_name=file_name,
        uploaded_by=user.id,
        status="received",
        target_domain=effective_domain,
        org_id=user_org_id(user),
    )
    db.add(upload)
    await db.flush()

    upload.s3_key = FileStorage().save(make_key(file_name), data)

    if is_large:
        # Persist pending marker and offload heavy work
        upload.error_report = {
            "target_domain": effective_domain,
            "status": "processing",
            "message": "large file accepted — processing in background. Poll GET /uploads/{id} for completion.",
            "file_size": len(data),
            "detected": detection,
        }
        await db.commit()
        await db.refresh(upload)
        background_tasks.add_task(
            _process_upload_background,
            upload.id,
            effective_domain,
            data,
            file_name,
            user_org_id(user),
            data_source_id,
        )
        out = UploadOut.model_validate(upload)
        return out

    try:
        extract = extract_tabular(data, file_name)
    except ValueError as e:
        upload.status = "failed"
        upload.error_report = {"error": str(e), "target_domain": effective_domain}
        await db.commit()
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e)) from e

    upload.status = "validated"
    upload.row_count = len(extract.frame)

    # If row count is also large, switch to background after validation to keep request snappy
    if upload.row_count and upload.row_count > ASYNC_THRESHOLD_ROWS:
        upload.status = "received"
        upload.error_report = _report(
            upload=upload,
            target_domain=effective_domain,
            kind=extract.kind,
            encoding=extract.encoding,
            columns=extract.columns,
            preview=extract.preview,
            warnings=extract.warnings,
            extra={
                "status": "processing",
                "message": f"{upload.row_count} rows — processing in background",
                "file_size": len(data),
            },
        )
        await db.commit()
        await db.refresh(upload)
        # Re-run via background using original data (avoids double extraction cost)
        background_tasks.add_task(
            _process_upload_background,
            upload.id,
            effective_domain,
            data,
            file_name,
            user_org_id(user),
            data_source_id,
        )
        out = UploadOut.model_validate(upload)
        return out

    try:
        result = await run_frame_pipeline(
            db, effective_domain, extract.frame, trigger="upload", source_id=data_source_id, org_id=user_org_id(user)
        )
    except ValueError as e:  # e.g. missing required columns
        upload.status = "failed"
        upload.error_report = _report(
            upload=upload,
            target_domain=effective_domain,
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
        target_domain=effective_domain,
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
            "detected": detection,
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
    total = (await db.execute(select(func.count()).select_from(stmt.subquery()))).scalar_one() or 0
    # Most recent first so the just-uploaded file appears at top without refresh
    stmt = stmt.order_by(RawUpload.created_at.desc())
    rows = (await db.execute(stmt.offset((page - 1) * page_size).limit(page_size))).scalars().all()
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
            select(RawUpload).where(RawUpload.id == upload_id, org_predicate(RawUpload.org_id, user.org_id))
        )
    ).scalar_one_or_none()
    if upload is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Upload not found")
    return UploadOut.model_validate(upload)
