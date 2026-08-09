
from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel

from app.api.deps import DbSession, get_current_user, require_role
from app.core.clock import business_today

router = APIRouter(
    prefix="/recommendations",
    tags=["recommendations"],
    dependencies=[Depends(get_current_user)],
)


class RecommendationOut(BaseModel):
    title: str
    body: str
    insight_type: str
    severity: str
    evidence: dict | None = None


@router.get("", response_model=list[RecommendationOut])
async def list_recommendations(
    db: DbSession,
    min_severity: str | None = Query(None, alias="min_severity"),
) -> list[RecommendationOut]:
    from app.services.ml.recommendations import generate_all_recommendations

    recs = await generate_all_recommendations(db)
    if min_severity:
        order = {"critical": 3, "warning": 2, "info": 1}
        threshold = order.get(min_severity, 0)
        recs = [r for r in recs if order.get(r["severity"], 0) >= threshold]
    return [RecommendationOut(**r) for r in recs]


@router.post(
    "/generate",
    dependencies=[Depends(require_role("manager"))],
    status_code=status.HTTP_200_OK,
)
async def generate_recommendations(
    db: DbSession,
) -> dict[str, int]:
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from app.models.decision import Insight
    from app.services.ml.recommendations import generate_all_recommendations

    recs = await generate_all_recommendations(db)
    created = 0
    for r in recs:
        payload = {
            **r,
            "period_start": business_today(),
            "period_end": business_today(),
        }
        stmt = (
            pg_insert(Insight)
            .values(**payload)
            .on_conflict_do_nothing(index_elements=["dedupe_key"])
        )
        result = await db.execute(stmt)
        if result.rowcount:
            created += 1
    await db.commit()
    return {"generated": len(recs), "new": created}
