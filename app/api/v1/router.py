from fastapi import APIRouter

from app.api.v1 import (
    admin,
    advanced,
    ai,
    analytics,
    audit,
    auth,
    compare,
    data_sources,
    decision,
    etl,
    health,
    landing,
    ml,
    quality,
    rbac,
    recommendations,
    uploads,
    users,
)

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(landing.router)
api_router.include_router(analytics.router)
api_router.include_router(advanced.router)
api_router.include_router(ml.router)
api_router.include_router(decision.router)
api_router.include_router(decision.manager_router)
api_router.include_router(decision.schedule_router)
api_router.include_router(auth.router)
api_router.include_router(users.router)
api_router.include_router(rbac.router)
api_router.include_router(audit.router)
api_router.include_router(data_sources.router)
api_router.include_router(uploads.router)
api_router.include_router(etl.router)
api_router.include_router(recommendations.router)
api_router.include_router(ai.router)
api_router.include_router(quality.router)
api_router.include_router(admin.router)
api_router.include_router(compare.router)
