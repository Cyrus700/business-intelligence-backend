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
) -> list[RecommendationOut]:
    from app.services.ml.recommendations import generate_all_recommendations, scope_recommendations

    recs = await generate_all_recommendations(db)
    recs = await scope_recommendations(db, recs, user)
    if min_severity:
        order = {"critical": 3, "warning": 2, "info": 1}
        threshold = order.get(min_severity, 0)
        recs = [r for r in recs if order.get(r["severity"], 0) >= threshold]
    return [RecommendationOut(**r) for r in recs]


@router.get("/history", response_model=list[RecommendationOut])
async def recommendation_history(db: DbSession) -> list[RecommendationOut]:
    """Persisted recommendations with their decision status (Phase 8 audit trail)."""
    rows = (
        (
            await db.execute(
                select(Insight)
                .where(Insight.insight_type == "recommendation")
                .order_by(Insight.generated_at.desc())
                .limit(100)
            )
        )
        .scalars()
        .all()
    )
    out = []
    for r in rows:
        out.append(
            RecommendationOut(
                id=r.id,
                title=r.title,
                body=r.body,
                insight_type=r.insight_type,
                severity=r.severity,
                evidence=r.evidence,
                dedupe_key=r.dedupe_key,
                impact_estimate=float(r.impact_estimate) if r.impact_estimate is not None else None,
                priority=r.priority,
                action=r.action,
                status=r.status,
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

    insight.status = body.decision
    if insight.dedupe_key:
        db.add(
            RecommendationFeedback(
                rec_key=insight.dedupe_key,
                user_id=user.id,
                action=body.decision,
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
        impact_estimate=float(insight.impact_estimate)
        if insight.impact_estimate is not None
        else None,
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
) -> dict[str, int]:
    from app.services.ml.recommendations import persist_recommendations

    return await persist_recommendations(db)
