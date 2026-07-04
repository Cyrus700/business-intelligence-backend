from fastapi import APIRouter

from app.api.v1 import (
    analytics,
    audit,
    auth,
    data_sources,
    etl,
    health,
    ml,
    stubs,
    uploads,
    users,
)

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(analytics.router)
api_router.include_router(ml.router)
api_router.include_router(auth.router)
api_router.include_router(users.router)
api_router.include_router(audit.router)
api_router.include_router(data_sources.router)
api_router.include_router(uploads.router)
api_router.include_router(etl.router)
api_router.include_router(stubs.router)
