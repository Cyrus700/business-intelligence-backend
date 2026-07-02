from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.audit import AuditMiddleware
from app.core.config import get_settings
from app.core.database import dispose_engine
from app.core.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    yield
    await dispose_engine()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="BI & Decision Support Dashboard API",
        version="0.1.0",
        description=(
            "Backend for the AI-Driven Cloud-Based Business Intelligence and "
            "Decision Support Dashboard (FYP — Sairash Budhathoki, NP069813)."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(AuditMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix="/api/v1")
    return app


app = create_app()
