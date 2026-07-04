from dataclasses import dataclass, field
from typing import Any


@dataclass
class RowError:
    row: int  # 1-based data row number (excluding header)
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"row": self.row, "reason": self.reason}


@dataclass
class TransformResult:
    """Outcome of validating + transforming one extracted batch."""

    records: list[dict[str, Any]] = field(default_factory=list)
    errors: list[RowError] = field(default_factory=list)

    @property
    def error_report(self) -> dict[str, Any]:
        return {
            "rejected": len(self.errors),
            "details": [e.as_dict() for e in self.errors[:200]],  # cap report size
        }


@dataclass
class LoadResult:
    loaded: int = 0
    skipped_duplicates: int = 0


@dataclass
class PipelineResult:
    job_id: str
    status: str
    rows_in: int
    rows_loaded: int
    rows_rejected: int
    skipped_duplicates: int
    error_report: dict[str, Any]
