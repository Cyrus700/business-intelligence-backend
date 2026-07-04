from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

SourceKind = Literal["csv_upload", "excel_upload", "rest_api", "postgres"]
TargetDomain = Literal["sales", "finance", "inventory"]


class DataSourceIn(BaseModel):
    name: str
    kind: SourceKind
    target_domain: TargetDomain
    config: dict[str, Any] = {}
    schedule_cron: str | None = None


class DataSourceUpdate(BaseModel):
    name: str | None = None
    config: dict[str, Any] | None = None
    schedule_cron: str | None = None
    status: Literal["active", "paused"] | None = None


class DataSourceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    kind: SourceKind
    target_domain: TargetDomain
    config: dict[str, Any]
    schedule_cron: str | None
    status: str
    created_at: datetime


class UploadOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    file_name: str
    status: str
    row_count: int | None
    error_report: dict[str, Any] | None
    created_at: datetime
    etl_job_id: str | None = None


class EtlJobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    data_source_id: UUID | None
    trigger: str
    status: str
    started_at: datetime
    finished_at: datetime | None
    rows_in: int | None
    rows_loaded: int | None
    rows_rejected: int | None
    log: dict[str, Any] | None
