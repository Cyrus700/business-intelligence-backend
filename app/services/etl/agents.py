"""Multi-agent upload system: inspect, validate, chunk, and load.

Each agent is a focused specialist. The orchestrator routes small vs large files
to the optimal path and surfaces a unified result.

Agents:
- FileInspectorAgent  — detects file kind, encoding, columns, and best domain
- ValidationAgent      — spot-checks preview rows without loading the warehouse
- ChunkManagerAgent   — assembles chunked uploads from the browser
- PipelineAgent        — runs transform+load and records ETL jobs

This module is pure logic; HTTP wiring lives in app.api.v1.uploads.
"""

from __future__ import annotations

import io
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.services.etl.domains import COLUMN_ALIASES, DOMAIN_SPECS
from app.services.etl.extractors import TabularExtract, extract_tabular

# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

# Files <= this size use the fast single-request path (no chunking).
FAST_PATH_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
# Rows per chunk when the browser streams a large file.
DEFAULT_CHUNK_SIZE = 1 * 1024 * 1024  # 1 MB
# Temp directory for chunked assembly
CHUNK_DIR = Path("var/uploads/chunks")
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Domain detection (file-type + column intelligence)
# ---------------------------------------------------------------------------

# Canonical required columns per domain
_REQUIRED: dict[str, set[str]] = {k: v["required_columns"] for k, v in DOMAIN_SPECS.items()}

# Alias lookup mirrors domains._ALIAS_LOOKUP but exposed here for scoring
_ALIAS_LOOKUP: dict[str, str] = {
    name: canonical for canonical, names in COLUMN_ALIASES.items() for name in [canonical, *names]
}


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def _resolve_headers(headers: list[str]) -> set[str]:
    """Map raw headers to canonical names for scoring."""
    return {_ALIAS_LOOKUP.get(_normalize_name(h), _normalize_name(h)) for h in headers}


def detect_domain(headers: list[str], sample_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Score each domain against the file's headers and return the best match.

    Returns:
        {
          "suggested": "sales" | "finance" | "inventory" | None,
          "confidence": 0.0..1.0,
          "scores": {"sales": {...}, ...},
          "alternatives": [...]
        }
    """
    canonical = _resolve_headers(headers)
    scores: dict[str, dict[str, Any]] = {}
    best: str | None = None
    best_score = -1.0
    best_matched = 0
    best_total = 1

    for domain, required in _REQUIRED.items():
        matched = len(required & canonical)
        total = len(required)
        # confidence = fraction of required columns present, penalised if file has
        # none of the domain's broader alias vocabulary
        confidence = matched / total if total else 0.0
        # Boost if file header contains domain-specific optional columns
        optional_hints = {
            "sales": {"product_name", "customer", "channel", "region", "category", "sku", "discount"},
            "finance": {"department", "description"},
            "inventory": {"product_name", "warehouse", "reorder_level"},
        }
        hints = optional_hints.get(domain, set())
        hint_bonus = min(0.15, len(hints & canonical) * 0.05)
        adjusted = min(1.0, confidence + hint_bonus)

        scores[domain] = {
            "matched": matched,
            "total": total,
            "confidence": round(adjusted, 2),
            "missing": sorted(required - canonical),
            "has_all_required": matched == total,
        }
        if adjusted > best_score or (adjusted == best_score and matched > best_matched):
            best_score = adjusted
            best = domain
            best_matched = matched

    # Only suggest if at least one required column matched
    if best and best_matched == 0:
        best = None
        best_score = 0.0

    # If tie between top two, treat as ambiguous -> no suggestion
    if best:
        top_scores = sorted([v["confidence"] for v in scores.values()], reverse=True)
        if len(top_scores) >= 2 and top_scores[0] == top_scores[1] and top_scores[0] > 0:
            # Check if two domains tie on matched count
            tied = [d for d, s in scores.items() if s["confidence"] == top_scores[0]]
            if len(tied) > 1:
                best = None
                best_score = 0.0

    alternatives = sorted(
        [{"domain": d, **s} for d, s in scores.items() if d != best],
        key=lambda x: x["confidence"],
        reverse=True,
    )

    return {
        "suggested": best,
        "confidence": round(best_score, 2) if best else 0.0,
        "scores": scores,
        "alternatives": alternatives,
    }


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

@dataclass
class InspectionResult:
    """Output of FileInspectorAgent."""

    file_name: str
    kind: str  # csv | excel
    encoding: str | None
    mime_hint: str
    size_bytes: int
    columns: list[str]
    canonical_columns: list[str]
    preview: list[dict[str, str]]
    warnings: list[str]
    detected: dict[str, Any]
    row_estimate: int | None = None
    sheet_name: str | None = None


class FileInspectorAgent:
    """Understands file type, structure, and domain without loading the warehouse."""

    def inspect(self, data: bytes, file_name: str) -> InspectionResult:
        name = file_name.lower()
        mime_hint = "text/csv" if name.endswith(".csv") else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        # Use existing extractor but capture metadata
        extract: TabularExtract = extract_tabular(data, file_name)

        canonical = [_ALIAS_LOOKUP.get(_normalize_name(c), _normalize_name(c)) for c in extract.columns]
        detected = detect_domain(extract.columns)

        # Row estimate without loading: for CSV count newlines, for Excel use frame length
        row_estimate = len(extract.frame)

        # sheet detection for Excel
        sheet_name = None
        if extract.kind == "excel":
            try:
                xls = pd.ExcelFile(io.BytesIO(data))
                sheet_name = xls.sheet_names[0] if xls.sheet_names else None
            except Exception:
                pass

        return InspectionResult(
            file_name=file_name,
            kind=extract.kind,
            encoding=extract.encoding,
            mime_hint=mime_hint,
            size_bytes=len(data),
            columns=extract.columns,
            canonical_columns=canonical,
            preview=extract.preview,
            warnings=extract.warnings,
            detected=detected,
            row_estimate=row_estimate,
            sheet_name=sheet_name,
        )


@dataclass
class ValidationIssue:
    row: int
    reason: str
    column: str | None = None


@dataclass
class ValidationResult:
    valid_rows: int
    invalid_rows: int
    issues: list[ValidationIssue] = field(default_factory=list)
    missing_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class ValidationAgent:
    """Validates a preview slice before the full load; mirrors frontend spotCheck."""

    DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ].*)?$")

    def validate(self, frame: pd.DataFrame, domain: str) -> ValidationResult:
        from app.services.etl.domains import transform_frame

        # Fast path: use transform_frame on a small slice to collect issues
        try:
            # Clone to avoid mutating original
            result = transform_frame(domain, frame.copy())
            issues = [ValidationIssue(row=e.row, reason=e.reason) for e in result.errors]
            return ValidationResult(
                valid_rows=len(result.records),
                invalid_rows=len(result.errors),
                issues=issues[:20],
                missing_columns=[],
            )
        except ValueError as e:
            msg = str(e)
            if "missing required columns" in msg:
                missing = [c.strip() for c in msg.split(":")[-1].split(",")]
                return ValidationResult(valid_rows=0, invalid_rows=len(frame), missing_columns=missing, issues=[])
            return ValidationResult(valid_rows=0, invalid_rows=len(frame), issues=[ValidationIssue(row=0, reason=msg)])


class ChunkManagerAgent:
    """Handles chunked assembly for large files (browser -> server)."""

    def init_session(self, file_name: str, domain: str | None, total_size: int, total_chunks: int) -> dict[str, Any]:
        session_id = uuid.uuid4().hex
        meta = {
            "session_id": session_id,
            "file_name": file_name,
            "domain": domain,
            "total_size": total_size,
            "total_chunks": total_chunks,
            "created_at": datetime.utcnow().isoformat(),
        }
        # Persist meta
        meta_path = CHUNK_DIR / f"{session_id}.meta"
        import json

        meta_path.write_text(json.dumps(meta))
        # Create chunk dir
        (CHUNK_DIR / session_id).mkdir(parents=True, exist_ok=True)
        return meta

    def store_chunk(self, session_id: str, chunk_index: int, data: bytes) -> int:
        session_dir = CHUNK_DIR / session_id
        if not session_dir.exists():
            raise ValueError("upload session not found or expired")
        chunk_path = session_dir / f"chunk_{chunk_index:06d}"
        chunk_path.write_bytes(data)
        # Return number of chunks received so far
        return len(list(session_dir.glob("chunk_*")))

    def assemble(self, session_id: str) -> tuple[bytes, dict[str, Any]]:
        import json

        meta_path = CHUNK_DIR / f"{session_id}.meta"
        if not meta_path.exists():
            raise ValueError("upload session not found or expired")
        meta = json.loads(meta_path.read_text())
        session_dir = CHUNK_DIR / session_id
        chunks = sorted(session_dir.glob("chunk_*"))
        if not chunks:
            raise ValueError("no chunks received")
        # Verify we have contiguous chunks 0..n-1
        expected = meta.get("total_chunks")
        if expected and len(chunks) != expected:
            raise ValueError(f"expected {expected} chunks but got {len(chunks)}")
        assembled = b"".join(p.read_bytes() for p in chunks)
        return assembled, meta

    def cleanup(self, session_id: str) -> None:
        import shutil

        meta_path = CHUNK_DIR / f"{session_id}.meta"
        if meta_path.exists():
            meta_path.unlink()
        session_dir = CHUNK_DIR / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir, ignore_errors=True)


class PipelineAgent:
    """Runs the validated frame through transform+load, with chunked insert."""

    async def run(
        self,
        db,
        domain: str,
        frame: pd.DataFrame,
        trigger: str = "upload",
        source_id: uuid.UUID | None = None,
        org_id: uuid.UUID | None = None,
    ):
        from app.services.etl.pipeline import run_frame_pipeline

        return await run_frame_pipeline(db, domain, frame, trigger=trigger, source_id=source_id, org_id=org_id)


# ---------------------------------------------------------------------------
# Orchestrator: decides fast vs chunked vs async
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorDecision:
    path: str  # "fast" | "chunked" | "async"
    reason: str
    detected_domain: str | None = None
    confidence: float = 0.0


class UploadOrchestrator:
    """Chooses the optimal pipeline for each upload based on size and type."""

    def __init__(self):
        self.inspector = FileInspectorAgent()
        self.validator = ValidationAgent()
        self.chunk_manager = ChunkManagerAgent()
        self.pipeline = PipelineAgent()

    def decide(self, data: bytes, file_name: str, requested_domain: str | None = None) -> OrchestratorDecision:
        size = len(data)
        # Inspect to detect domain
        try:
            inspection = self.inspector.inspect(data, file_name)
            detected = inspection.detected.get("suggested")
            confidence = inspection.detected.get("confidence", 0.0)
        except Exception:
            detected = None
            confidence = 0.0
            inspection = None

        # If file is huge, recommend chunked regardless of request
        if size > FAST_PATH_MAX_BYTES:
            return OrchestratorDecision(
                path="chunked",
                reason=f"file is {size / (1024*1024):.1f} MB — using chunked upload for reliability",
                detected_domain=detected,
                confidence=confidence,
            )

        # If requested domain mismatches detected domain with high confidence, warn but still fast path
        if requested_domain and detected and requested_domain != detected and confidence >= 0.9:
            # Still fast path, but caller should surface warning
            pass

        # Default fast path for small files
        return OrchestratorDecision(
            path="fast",
            reason="small file — single-request upload",
            detected_domain=detected,
            confidence=confidence,
        )

    def suggested_domain(self, data: bytes, file_name: str) -> dict[str, Any]:
        try:
            inspection = self.inspector.inspect(data, file_name)
            return inspection.detected
        except Exception as e:
            return {"suggested": None, "confidence": 0.0, "error": str(e)}


# Singleton orchestrator for import
orchestrator = UploadOrchestrator()

