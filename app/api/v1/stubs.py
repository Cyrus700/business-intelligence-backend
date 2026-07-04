"""Placeholder routers for later phases. Each returns 501 until its phase lands.

Phase 2: data-sources, uploads, etl · Phase 3: kpis, sales, finance, inventory
Phase 4: forecasts, anomalies, trends · Phase 5: insights, alert-rules, notifications, reports
"""

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_current_user

PLANNED = {
    "insights": 5,
    "alert-rules": 5,
    "notifications": 5,
    "reports": 5,
}

router = APIRouter(dependencies=[Depends(get_current_user)])


def _make_stub(prefix: str, phase: int):
    async def stub() -> None:
        raise HTTPException(
            status.HTTP_501_NOT_IMPLEMENTED,
            f"/{prefix} is planned for Phase {phase} (see docs/plan/)",
        )

    return stub


for _prefix, _phase in PLANNED.items():
    router.add_api_route(
        f"/{_prefix}",
        _make_stub(_prefix, _phase),
        methods=["GET"],
        tags=["planned"],
        name=f"stub_{_prefix.replace('-', '_')}",
    )
