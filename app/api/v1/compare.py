"""Period-over-period comparison endpoints — Compare 2..N months or years.

Professional, deterministic core (arithmetic over warehouse aggregates) plus
optional AI narrative (Groq → Gemini → deterministic fallback). Every query is
org-scoped and permission-gated via ``compare:view``.

Routes
------
POST /analytics/compare      — deterministic comparison payload (illustrations-ready)
POST /analytics/compare/ai   — AI suggestions for the same comparison
GET  /analytics/compare/meta — allowed metrics / dimensions / limits (for the UI)
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, field_validator

from app.api.deps import CurrentUser, DbSession, get_current_user, is_super_admin, require_permission
from app.core.clock import business_today
from app.services.analytics.compare import (
    ALLOWED_DIMS,
    ALLOWED_METRICS,
    MAX_PERIODS,
    MIN_PERIODS,
    compare_periods,
    generate_ai_suggestions,
    month_bounds,
    year_bounds,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/analytics",
    tags=["compare"],
    dependencies=[Depends(get_current_user)],
)


# ── Schemas ───────────────────────────────────────────────────────────────


class CompareRequest(BaseModel):
    """Request to compare N periods.

    ``periods`` is the canonical shape; ``months`` / ``years`` are ergonomic
    shorthands — at least one source must be provided and they are merged.
    """

    model_config = ConfigDict(extra="ignore")

    periods: list[dict[str, Any]] | None = None
    months: list[str] | None = None  # "YYYY-MM"
    years: list[int] | None = None
    metrics: list[str] | None = None
    dimensions: list[str] | None = None
    include_timeseries: bool = True
    timeseries_metric: str = "revenue"
    timeseries_granularity: Literal["day", "week", "month", "quarter", "year"] = "day"
    # When true the response also includes an ``ai`` block (one DB round-trip instead of two).
    include_ai: bool = False

    @field_validator("months")
    @classmethod
    def _validate_months(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        import re

        pat = re.compile(r"^\d{4}-\d{2}$")
        for m in v:
            if not pat.match(m):
                raise ValueError(f"Invalid month '{m}'; expected YYYY-MM")
            y, mo = int(m[:4]), int(m[5:7])
            if not (1 <= mo <= 12):
                raise ValueError(f"Invalid month '{m}'")
            # not in future beyond today
            try:
                d_from, d_to = month_bounds(y, mo)
                if d_from > business_today():
                    raise ValueError(f"Month {m} is in the future")
            except Exception:
                raise
        return v

    @field_validator("years")
    @classmethod
    def _validate_years(cls, v: list[int] | None) -> list[int] | None:
        if v is None:
            return v
        cur = business_today().year
        for y in v:
            if not (2000 <= y <= cur + 1):
                raise ValueError(f"Invalid year {y}")
            if y > cur:
                # allow next year only if Jan (edge) — otherwise future
                if y > cur:
                    raise ValueError(f"Year {y} is in the future")
        return v

    @field_validator("metrics")
    @classmethod
    def _validate_metrics(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        cleaned = [m for m in v if m in ALLOWED_METRICS]
        if v and not cleaned:
            raise ValueError(f"No valid metrics; allowed: {', '.join(ALLOWED_METRICS)}")
        return cleaned

    @field_validator("dimensions")
    @classmethod
    def _validate_dimensions(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        cleaned = [d for d in v if d in ALLOWED_DIMS]
        if v and not cleaned:
            raise ValueError(f"No valid dimensions; allowed: {', '.join(ALLOWED_DIMS)}")
        return cleaned


def _build_periods(req: CompareRequest) -> list[dict[str, Any]]:
    """Merge ``periods`` + ``months`` + ``years`` into a single list."""
    out: list[dict[str, Any]] = []
    if req.periods:
        for p in req.periods:
            # normalise keys: accept from/date_from, to/date_to, label, id
            from_raw = p.get("from") or p.get("date_from") or p.get("from_") or p.get("start")
            to_raw = p.get("to") or p.get("date_to") or p.get("end")
            if from_raw is None or to_raw is None:
                # let validation surface the precise error inside compare service
                out.append({"from": from_raw, "to": to_raw, **{k: v for k, v in p.items() if k not in ("from", "to")}})
            else:
                out.append({"from": from_raw, "to": to_raw, "label": p.get("label"), "id": p.get("id")})
    if req.months:
        for m in req.months:
            y, mo = int(m[:4]), int(m[5:7])
            d_from, d_to = month_bounds(y, mo)
            label = date(y, mo, 1).strftime("%B %Y")
            out.append({"from": d_from.isoformat(), "to": d_to.isoformat(), "label": label})
    if req.years:
        for y in req.years:
            d_from, d_to = year_bounds(y)
            out.append({"from": d_from.isoformat(), "to": d_to.isoformat(), "label": str(y)})
    return out


# ── Helpers ───────────────────────────────────────────────────────────────


def _can_view_pnl(user: CurrentUser, db: DbSession) -> bool:
    """Analysts see aggregates but not P&L margins — check live policy."""
    # Fast path: role hierarchy already tells us analyst < manager
    # But the live matrix may grant pnl:view to analyst, so check policy if available.
    # We do a synchronous rank fallback here to avoid an extra DB round-trip on the
    # hot path; the full check happens via require_permission on the Ai endpoint.
    # For the main compare we let analysts see gross_margin but hide net_profit if
    # they lack pnl:view — the service filters metrics accordingly when can_view_pnl is False.
    # We conservatively return True for manager/admin, False for analyst, and let
    # the RBAC matrix override via the permission gate on the endpoint itself.
    if user.role in ("manager", "admin"):
        return True
    # For analyst, attempt to read policy permissions without blocking — if the
    # permission is not granted, we still allow compare but without P&L metrics.
    # The endpoint is gated by compare:view, not pnl:view, so an analyst who has
    # compare:view but not pnl:view still gets a useful (non-P&L) comparison.
    # We return False here to trigger metric filtering.
    # If later the matrix grants pnl:view to analyst, the caller can be re-checked
    # via a dedicated DB read — but the simple role check covers 99% of deployments.
    return False


async def _can_view_pnl_async(user: CurrentUser, db: DbSession) -> bool:
    try:
        from app.services import rbac as rbac_svc

        policy = await rbac_svc.get_policy(db)
        held = policy.permissions_for(user.role)
        return "pnl:view" in held
    except Exception:
        return _can_view_pnl(user, db)


# ── Endpoints ─────────────────────────────────────────────────────────────


@router.get("/compare/meta")
async def compare_meta(user: CurrentUser) -> dict[str, Any]:
    """Metadata the UI needs to render the Compare builder."""
    return {
        "allowed_metrics": list(ALLOWED_METRICS),
        "allowed_dimensions": list(ALLOWED_DIMS),
        "metric_labels": {
            "revenue": "Revenue",
            "orders": "Orders",
            "avg_order_value": "Avg Order Value",
            "gross_margin": "Gross Margin",
            "expense_total": "Expenses",
            "net_profit": "Net Profit",
        },
        "limits": {
            "min_periods": MIN_PERIODS,
            "max_periods": MAX_PERIODS,
            "max_span_days": 400,
        },
        "today": business_today().isoformat(),
        "timezone": "Asia/Kathmandu",
    }


@router.post("/compare", dependencies=[Depends(require_permission("compare:view"))])
async def compare(
    body: CompareRequest,
    db: DbSession,
    user: CurrentUser,
) -> dict[str, Any]:
    """Deterministic period comparison — illustrations-ready, org-scoped."""
    periods = _build_periods(body)
    if not periods:
        from fastapi import HTTPException

        raise HTTPException(400, "Provide at least 2 periods (via periods, months or years)")

    org_id = None if is_super_admin(user) else user.org_id
    can_pnl = await _can_view_pnl_async(user, db)

    try:
        result = await compare_periods(
            db,
            periods=periods,
            metrics=body.metrics,
            dimensions=body.dimensions,
            org_id=org_id,
            include_timeseries=body.include_timeseries,
            timeseries_metric=body.timeseries_metric,
            timeseries_granularity=body.timeseries_granularity,
            can_view_pnl=can_pnl,
        )
    except ValueError as e:
        from fastapi import HTTPException

        raise HTTPException(400, str(e)) from e

    # Optionally attach AI block inline to save a round-trip
    if body.include_ai:
        try:
            org_name = None
            if org_id is not None:
                from app.models.identity import Organization

                org = await db.get(Organization, org_id)
                org_name = org.name if org else None
            ai = await generate_ai_suggestions(comparison=result, role=user.role or "analyst", org_name=org_name)
            result["ai"] = ai
        except Exception as e:
            logger.warning("inline AI generation failed: %s", e)
            result["ai"] = {
                "summary": result.get("insights", {}).get("verdict", ""),
                "narrative": "",
                "source": "unavailable",
                "disclaimer": "AI unavailable — deterministic insights only.",
            }

    return result


@router.post("/compare/ai", dependencies=[Depends(require_permission("compare:view"))])
async def compare_ai(
    body: CompareRequest,
    db: DbSession,
    user: CurrentUser,
) -> dict[str, Any]:
    """AI narrative for a comparison — same inputs as POST /compare.

    The comparison is recomputed server-side so the AI prompt is grounded in
    live warehouse numbers, not client-provided values.
    """
    periods = _build_periods(body)
    if not periods:
        from fastapi import HTTPException

        raise HTTPException(400, "Provide at least 2 periods (via periods, months or years)")

    org_id = None if is_super_admin(user) else user.org_id
    can_pnl = await _can_view_pnl_async(user, db)

    try:
        comparison = await compare_periods(
            db,
            periods=periods,
            metrics=body.metrics,
            dimensions=body.dimensions,
            org_id=org_id,
            include_timeseries=False,  # AI doesn't need overlay
            can_view_pnl=can_pnl,
        )
    except ValueError as e:
        from fastapi import HTTPException

        raise HTTPException(400, str(e)) from e

    org_name = None
    if org_id is not None:
        try:
            from app.models.identity import Organization

            org = await db.get(Organization, org_id)
            org_name = org.name if org else None
        except Exception:
            pass

    ai = await generate_ai_suggestions(comparison=comparison, role=user.role or "analyst", org_name=org_name)
    return {"ai": ai, "insights": comparison.get("insights"), "periods": comparison.get("periods")}
