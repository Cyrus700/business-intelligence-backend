from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession, get_current_user, require_role
from app.models import Insight, RecommendationFeedback

router = APIRouter(
    prefix="/recommendations",
    tags=["recommendations"],
    dependencies=[Depends(get_current_user)],
)


class RecommendationOut(BaseModel):
    id: UUID | None = None
    title: str
    body: str
    insight_type: str
    severity: str
    evidence: dict | None = None
    dedupe_key: str | None = None
    impact_estimate: float | None = None
    impact_basis: str | None = None
    priority: str | None = None  # why-now: priority = impact × severity
    action: str | None = None  # recommended next step
    status: str | None = None  # open | accepted | dismissed | postponed | actioned


class DecisionBody(BaseModel):
    decision: str  # accepted | dismissed | postponed | actioned


@router.get("", response_model=list[RecommendationOut])
async def list_recommendations(
    db: DbSession,
    user: CurrentUser,
    min_severity: str | None = Query(None, alias="min_severity"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> list[RecommendationOut]:
    # validate min_severity
    if min_severity is not None and min_severity not in ("info", "warning", "critical"):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"invalid min_severity: {min_severity}")

    from app.services.analytics.cache import get_query_cache
    from app.services.ml.recommendations import generate_all_recommendations, scope_recommendations

    org_id = None if getattr(user, "is_super_admin", False) else user.org_id
    cache = get_query_cache()
    cache_key = cache._make_key("recommendations_list", (str(org_id), min_severity, user.role), {})  # type: ignore
    # try cache (short ttl 60s) — recommendations are deterministic per day
    cached = await cache.get(cache_key)
    if cached is not None:
        # cached is already scoped? store raw recs and re-scope per role? For simplicity cache scoped per role
        return [RecommendationOut(**r) for r in cached[offset: offset+limit]]

    recs = await generate_all_recommendations(db, org_id=org_id)
    recs = await scope_recommendations(db, recs, user)
    if min_severity:
        order = {"critical": 3, "warning": 2, "info": 1}
        threshold = order.get(min_severity, 0)
        recs = [r for r in recs if order.get(r["severity"], 0) >= threshold]
    # cache full list before pagination
    await cache.set(cache_key, recs, 60)
    paginated = recs[offset: offset+limit]
    return [RecommendationOut(**r) for r in paginated]


@router.get("/history", response_model=list[RecommendationOut])
async def recommendation_history(
    db: DbSession,
    user: CurrentUser,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> list[RecommendationOut]:
    """Persisted recommendations with their decision status (Phase 8 audit trail)."""
    from app.services.ml.recommendations import scope_recommendations

    stmt = select(Insight).where(Insight.insight_type == "recommendation")
    if not getattr(user, "is_super_admin", False):
        stmt = stmt.where(Insight.org_id == user.org_id)
    rows = (await db.execute(stmt.order_by(Insight.generated_at.desc()).limit(limit).offset(offset))).scalars().all()
    # build rec dicts for scoping
    recs = []
    for r in rows:
        recs.append(
            {
                "id": r.id,
                "title": r.title,
                "body": r.body,
                "insight_type": r.insight_type,
                "severity": r.severity,
                "evidence": r.evidence,
                "dedupe_key": r.dedupe_key,
                "impact_estimate": float(r.impact_estimate) if r.impact_estimate is not None else None,
                "priority": r.priority,
                "action": r.action,
                "status": r.status,
                # preserve kind if deducible
                "kind": (r.dedupe_key.split(":")[1] if r.dedupe_key and ":" in r.dedupe_key and len(r.dedupe_key.split(":")[0])==36 else r.dedupe_key.split(":")[0] if r.dedupe_key else None),
            }
        )
    # fix history redaction via scope_recommendations — analysts should not see sensitive bodies
    scoped = await scope_recommendations(db, recs, user)
    # map back preserving id/status
    # scope_recommendations may reorder by impact; keep that order
    out = []
    for rec in scoped:
        out.append(
            RecommendationOut(
                id=rec.get("id"),
                title=rec["title"],
                body=rec["body"],
                insight_type=rec["insight_type"],
                severity=rec["severity"],
                evidence=rec["evidence"],
                dedupe_key=rec["dedupe_key"],
                impact_estimate=float(rec["impact_estimate"]) if rec.get("impact_estimate") is not None else None,
                priority=rec.get("priority"),
                action=rec.get("action"),
                status=rec.get("status"),
            )
        )
    return out


@router.post(
    "/{insight_id}/decide",
    response_model=RecommendationOut,
    dependencies=[Depends(require_role("manager"))],
)
async def decide_recommendation(
    insight_id: UUID,
    body: DecisionBody,
    db: DbSession,
    user: CurrentUser,
) -> RecommendationOut:
    """Decision workflow: accept / dismiss / postpone / action a recommendation.

    The decision is persisted twice: on the insight (status, the auditable
    lifecycle) and in recommendation_feedback (rec_key aggregate, which later
    ranks what to surface first).
    """
    if body.decision not in ("accepted", "dismissed", "postponed", "actioned"):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "invalid decision")

    insight = await db.get(Insight, insight_id)
    if insight is None or insight.insight_type != "recommendation":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Recommendation not found")
    if not getattr(user, "is_super_admin", False) and insight.org_id != user.org_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Recommendation not found")

    insight.status = body.decision
    if insight.dedupe_key:
        # fix feedback key mismatch: store rec_key without org prefix for consistent aggregation
        raw_key = insight.dedupe_key
        # if deduped with org prefix like "{org_id}:{kind}:...", strip org prefix
        if insight.org_id and raw_key.startswith(str(insight.org_id) + ":"):
            raw_key = raw_key[len(str(insight.org_id)) + 1 :]
        db.add(
            RecommendationFeedback(
                rec_key=raw_key,
                user_id=user.id,
                action=body.decision,
                org_id=insight.org_id,
            )
        )
    await db.commit()
    await db.refresh(insight)
    return RecommendationOut(
        id=insight.id,
        title=insight.title,
        body=insight.body,
        insight_type=insight.insight_type,
        severity=insight.severity,
        evidence=insight.evidence,
        dedupe_key=insight.dedupe_key,
        impact_estimate=float(insight.impact_estimate) if insight.impact_estimate is not None else None,
        priority=insight.priority,
        action=insight.action,
        status=insight.status,
    )


@router.post(
    "/generate",
    dependencies=[Depends(require_role("manager"))],
    status_code=status.HTTP_200_OK,
)
async def generate_recommendations(
    db: DbSession,
    user: CurrentUser,
) -> dict[str, int]:
    from app.services.ml.recommendations import persist_recommendations

    org_id = None if getattr(user, "is_super_admin", False) else user.org_id
    # bust cache after generation to reflect new persisted insights
    result = await persist_recommendations(db, org_id=org_id)
    try:
        from app.services.analytics.cache import clear_query_cache

        await clear_query_cache(org_id=org_id)
    except Exception:
        pass
    return result
