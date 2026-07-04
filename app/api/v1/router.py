from fastapi import APIRouter

from app.api.v1 import (
    analytics,
    audit,
    auth,
    data_sources,
    decision,
    etl,
    health,
    ml,
    uploads,
    users,
)

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(analytics.router)
api_router.include_router(ml.router)
api_router.include_router(decision.router)
api_router.include_router(decision.manager_router)
api_router.include_router(auth.router)
api_router.include_router(users.router)
api_router.include_router(audit.router)
api_router.include_router(data_sources.router)
api_router.include_router(uploads.router)
api_router.include_router(etl.router)
