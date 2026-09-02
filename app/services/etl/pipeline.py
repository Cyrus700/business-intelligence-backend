"""Pipeline orchestrator: extract → validate/transform → load → rebuild KPIs.

Every run is recorded in etl_jobs (the admin monitoring screen's data source),
including failures.
"""

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.request_context import current_request_id
from app.models import DataSource, EtlJob
from app.services.analytics.kpi_builder import rebuild_kpi_snapshots
from app.services.etl.base import PipelineResult
from app.services.etl.domains import transform_frame
from app.services.etl.extractors import extract_postgres, extract_rest_api
from app.services.etl.loader import LOADERS
from app.services.etl.refresh import refresh_derived

logger = logging.getLogger(__name__)

_DATE_FIELDS = {"sales": "txn_date", "finance": "expense_date", "inventory": "snapshot_date"}


async def run_frame_pipeline(
    db: AsyncSession,
    domain: str,
    frame: pd.DataFrame,
    *,
    trigger: str,
    source_id: uuid.UUID | None = None,
    org_id: uuid.UUID | None = None,
) -> PipelineResult:
    """Run the transform+load stages on an already-extracted DataFrame."""
    # org_id is trusted from current_user, never client-supplied
    if org_id is None:
        raise ValueError("Organization context required for business data — your account has no workspace. Register a business or request an invite.")
    # If source_id is given, org_id must match the source's org (enforced by caller)
    if source_id is not None:
        src = await db.get(DataSource, source_id)
        if src and src.org_id != org_id:
            raise ValueError("Source does not belong to caller's organization")
    job = EtlJob(
        data_source_id=source_id,
        trigger=trigger,
        rows_in=len(frame),
        log={"correlation_id": current_request_id()},
        org_id=org_id,
    )
    db.add(job)
    await db.flush()

    try:
        result = transform_frame(domain, frame)
        loader = LOADERS[domain]
        load_result = await loader(db, result.records, source_id, job.id, org_id=org_id)

        if result.records:
            date_field = _DATE_FIELDS[domain]
            dates = [r[date_field] for r in result.records]
            await rebuild_kpi_snapshots(db, min(dates), max(dates), org_id=org_id)

        job.status = "succeeded"
        job.rows_loaded = load_result.loaded
        job.rows_rejected = len(result.errors)
        job.log = {
            "skipped_duplicates": load_result.skipped_duplicates,
            "error_report": result.error_report,
        }
    except Exception as e:
        logger.exception("ETL job %s failed", job.id)
        job.status = "failed"
        job.rows_loaded = 0
        job.rows_rejected = 0
        job.log = {"error": str(e)}
        job.finished_at = datetime.now(UTC).replace(tzinfo=None)
        await db.commit()
        raise

    job.finished_at = datetime.now(UTC).replace(tzinfo=None)
    # Preserve values before commit expiry (SQLAlchemy expires on commit)
    _job_id = str(job.id)
    _base_log = dict(job.log or {})
    _rows_in = job.rows_in
    _rows_loaded = job.rows_loaded
    _rows_rejected = job.rows_rejected
    await db.commit()

    # The rows are committed and the job is recorded, so from here nothing may
    # fail the ingest. Bringing anomalies, insights and the assistant's index
    # forward now is what makes an upload visible everywhere at once instead of
    # only in the KPI cards.
    if result.records:
        date_field = _DATE_FIELDS[domain]
        dates = [r[date_field] for r in result.records]
        refresh = await refresh_derived(db, min(dates), max(dates))
        # Re-attach job instance after expiry to update log safely
        try:
            await db.refresh(job)
            job.log = {**_base_log, "post_load_refresh": refresh.as_log()}
            await db.commit()
            _base_log = dict(job.log or {})
        except Exception:
            logger.warning("could not record post-load refresh on job %s", _job_id, exc_info=True)
            try:
                await db.rollback()
            except Exception:
                pass

    return PipelineResult(
        job_id=_job_id,
        status="succeeded",
        rows_in=_rows_in or 0,
        rows_loaded=_rows_loaded or 0,
        rows_rejected=_rows_rejected or 0,
        skipped_duplicates=_base_log.get("skipped_duplicates", 0),
        error_report=_base_log.get("error_report", {"details": []}),
    )


async def run_source_pipeline(
    db: AsyncSession, source: DataSource, *, trigger: str, org_id: uuid.UUID | None = None
) -> PipelineResult:
    """Pull-based run for rest_api / postgres sources."""
    effective_org = org_id if org_id is not None else source.org_id
    config: dict[str, Any] = source.config or {}
    if source.kind == "rest_api":
        frame = await extract_rest_api(config)
    elif source.kind == "postgres":
        frame = await extract_postgres(config)
    else:
        raise ValueError(f"source kind '{source.kind}' is push-based; use the upload endpoint")
    return await run_frame_pipeline(
        db, source.target_domain, frame, trigger=trigger, source_id=source.id, org_id=effective_org
    )
